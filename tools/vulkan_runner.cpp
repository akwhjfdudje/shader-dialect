// vulkan_runner.cpp — Load SPIR-V fragment shader and execute on GPU via Vulkan.
//
// Usage:  vulkan_runner.exe <vertex.spv> <fragment.spv> [entry_point_name]
//
// Creates a minimal graphics pipeline (fullscreen triangle) using the provided
// vertex and fragment shaders, sets up dummy descriptor resources, renders to a
// 1x1 offscreen framebuffer, and prints the resulting pixel value(s).
//
// Compile with MSVC (from a Developer PowerShell / VS tools environment):
//   cl /std:c++17 /EHsc /O2 /I "%VULKAN_SDK%\Include" ^
//      vulkan_runner.cpp /link "%VULKAN_SDK%\Lib\vulkan-1.lib"

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal SPIR-V binary reflection helpers
// ---------------------------------------------------------------------------

struct SpvModule {
    const uint32_t *words = nullptr;
    uint32_t size = 0;
    uint32_t bound = 0;  // highest ID

    // Parsed reflection data
    std::string entryName;          // first OpEntryPoint name
    uint32_t entryFuncId = 0;       // entry point function ID
    std::vector<uint32_t> texBindings;   // binding indices for sampled images
    std::vector<uint32_t> sampBindings;  // binding indices for samplers
    uint32_t maxOutputLocation = 0; // max fragment output location + 1
};

// Read a SPIR-V binary file into a vector, returns empty on failure.
static std::vector<uint32_t> readSpvFile(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        return {};
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> words(sz / 4 + (sz % 4 ? 1 : 0));
    fread(words.data(), 1, sz, f);
    fclose(f);
    return words;
}

// Reflect a SPIR-V binary to extract entry point name and descriptor bindings.
// Looks for OpEntryPoint, OpDecorate (DescriptorSet/Binding), and type info.
static bool reflectSpv(const std::vector<uint32_t> &spv, SpvModule &m) {
    m.words = spv.data();
    m.size = static_cast<uint32_t>(spv.size());
    if (m.size < 5 || spv[0] != 0x07230203) {
        fprintf(stderr, "ERROR: not a valid SPIR-V binary\n");
        return false;
    }
    m.bound = spv[3];

    uint32_t i = 5;  // skip header
    while (i < m.size) {
        uint32_t wc = spv[i] >> 16;
        uint32_t op = spv[i] & 0xFFFF;
        if (wc == 0) { ++i; continue; }

        if (op == 15) {  // OpEntryPoint
            // [exec_model, func_id, name, interface...]
            if (m.entryName.empty()) {
                m.entryFuncId = spv[i + 2];
                m.entryName = reinterpret_cast<const char *>(&spv[i + 3]);
            }
        } else if (op == 71 && wc >= 3) {  // OpDecorate
            uint32_t deco = spv[i + 2];
            if (deco == 34 && wc >= 4) {    // DescriptorSet
                // spv[i+1] = target var, spv[i+3] = set
                uint32_t set = spv[i + 3];
                (void)set;  // we assume set 0 for now
            } else if (deco == 33 && wc >= 4) {  // Binding
                // Need to know if this variable is an image or sampler.
                // We'll resolve later by looking at the variable's type.
                // For now, record binding index for later type resolution.
            }
        }
        i += wc;
    }

    // Second pass: find GlobalVariables decorated with Binding, resolve
    // their pointer type to see if they're images or samplers.
    // We build a map: result_id -> opcode for type resolution.
    struct VarInfo { uint32_t binding; uint32_t typeId; };
    std::vector<VarInfo> vars;  // indexed by result_id (using a map)

    // Use a simple dynamic array indexed by result ID
    std::vector<uint32_t> resultOp(spv[3], 0);  // opcode for each result_id
    std::vector<uint32_t> varType(spv[3], 0);   // type_id for OpVariable results
    std::vector<uint32_t> varBinding(spv[3], ~0u);

    i = 5;
    while (i < m.size) {
        uint32_t wc = spv[i] >> 16;
        uint32_t op = spv[i] & 0xFFFF;
        if (wc == 0) { ++i; continue; }

        if (wc >= 2 && op < spv[3])
            resultOp[op] = op;  // not useful, but keep for future

        if (op == 59 && wc >= 4) {  // OpVariable
            uint32_t resultId = spv[i + 2];
            if (resultId < spv[3]) {
                varType[resultId] = spv[i + 1];  // result type
            }
        } else if (op == 71 && wc >= 4) {  // OpDecorate
            uint32_t target = spv[i + 1];
            uint32_t deco = spv[i + 2];
            if (deco == 33 && target < spv[3]) {  // Binding
                varBinding[target] = spv[i + 3];
            }
        }
        i += wc;
    }

    // Now resolve: for each variable that has a binding, trace its type
    // to determine if it's an image or sampler.
    // Type instructions store their result_id at word position 1.
    i = 5;
    while (i < m.size) {
        uint32_t wc = spv[i] >> 16;
        uint32_t op = spv[i] & 0xFFFF;
        if (wc == 0) { ++i; continue; }
        // Record type definitions: result_id = spv[i+1] if wc >= 2
        if (wc >= 2) {
            uint32_t tid = spv[i + 1];  // type result ID (for OpType*)
            if (tid < spv[3] && op >= 19 && op <= 42) {
                // Store type opcode for later resolution
            }
        }
        i += wc;
    }

    // For now, use a simpler approach: iterate decorations and use
    // type tracing to classify bindings.
    // Actually, let's just hardcode the classification based on the
    // known WrapSPIRVModule output: textures come first, samplers follow.
    // We'll count all binding decorations and classify by type.

    // Final pass: collect bindings with type classification
    i = 5;
    std::vector<uint32_t> imgTypes, sampTypes;
    while (i < m.size) {
        uint32_t wc = spv[i] >> 16;
        uint32_t op = spv[i] & 0xFFFF;
        if (wc == 0) { ++i; continue; }

        if (op == 25 && wc >= 2) {    // OpTypeImage — result_id at i+1
            imgTypes.push_back(spv[i + 1]);
        } else if (op == 26 && wc >= 2) {  // OpTypeSampler
            sampTypes.push_back(spv[i + 1]);
        } else if (op == 27 && wc >= 2) {  // OpTypeSampledImage
            // sampled images are derived from images;
            // the result type is an image type, we'll handle this via pointers
        }
        i += wc;
    }

    // Now walk all variables with bindings, trace pointer type ->
    // storage class, and classify.
    i = 5;
    while (i < m.size) {
        uint32_t wc = spv[i] >> 16;
        uint32_t op = spv[i] & 0xFFFF;
        if (wc == 0) { ++i; continue; }

        if (op == 59 && wc >= 4) {  // OpVariable
            uint32_t resultId = spv[i + 2];
            uint32_t binding = (resultId < spv[3]) ? varBinding[resultId] : ~0u;
            if (binding != ~0u) {
                uint32_t ptrTypeId = spv[i + 1];
                // Find the pointer type: OpTypePointer result_id, storage_class, pointee_type
                uint32_t pointeeTypeId = 0;
                uint32_t j = 5;
                while (j < m.size) {
                    uint32_t wc2 = spv[j] >> 16;
                    uint32_t op2 = spv[j] & 0xFFFF;
                    if (op2 == 32 && wc2 >= 4 && spv[j + 1] == ptrTypeId) {
                        pointeeTypeId = spv[j + 3];
                        break;
                    }
                    j += (wc2 ? wc2 : 1);
                }
                // Check if pointeeTypeId is an image type or sampler type
                bool isImage = (std::find(imgTypes.begin(), imgTypes.end(),
                                         pointeeTypeId) != imgTypes.end());
                bool isSampler = (std::find(sampTypes.begin(), sampTypes.end(),
                                            pointeeTypeId) != sampTypes.end());

                if (isImage)
                    m.texBindings.push_back(binding);
                else if (isSampler)
                    m.sampBindings.push_back(binding);
            }
        }
        i += wc;
    }

    // Sort bindings for consistency
    std::sort(m.texBindings.begin(), m.texBindings.end());
    std::sort(m.sampBindings.begin(), m.sampBindings.end());

    return !m.entryName.empty();
}

// ---------------------------------------------------------------------------
// Vulkan helpers
// ---------------------------------------------------------------------------

#define MAX_ATTACHMENTS 8  // max output locations supported

#define VK_CHECK(expr) do { VkResult r_ = (expr); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "Vulkan error %d at %s:%d\n", r_, __FILE__, __LINE__); \
    return 1; } } while(0)

struct VulkanState {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkCommandPool cmdPool = VK_NULL_HANDLE;

    VkDescriptorSetLayout dsLayout = VK_NULL_HANDLE;
    VkDescriptorPool dsPool = VK_NULL_HANDLE;
    VkDescriptorSet ds = VK_NULL_HANDLE;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkRenderPass renderPass = VK_NULL_HANDLE;

    uint32_t attachmentCount = 1;
    VkImage framebufImages[MAX_ATTACHMENTS] = {};
    VkDeviceMemory framebufMems[MAX_ATTACHMENTS] = {};
    VkImageView framebufViews[MAX_ATTACHMENTS] = {};
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
};

static void destroyVulkan(VulkanState &vk) {
    if (vk.device) {
        vkDeviceWaitIdle(vk.device);
        if (vk.pipeline)        vkDestroyPipeline(vk.device, vk.pipeline, nullptr);
        if (vk.pipelineLayout)  vkDestroyPipelineLayout(vk.device, vk.pipelineLayout, nullptr);
        if (vk.dsLayout)        vkDestroyDescriptorSetLayout(vk.device, vk.dsLayout, nullptr);
        if (vk.dsPool)          vkDestroyDescriptorPool(vk.device, vk.dsPool, nullptr);
        if (vk.renderPass)      vkDestroyRenderPass(vk.device, vk.renderPass, nullptr);
        if (vk.framebuffer)     vkDestroyFramebuffer(vk.device, vk.framebuffer, nullptr);
        for (uint32_t a = 0; a < vk.attachmentCount; ++a) {
            if (vk.framebufViews[a]) vkDestroyImageView(vk.device, vk.framebufViews[a], nullptr);
            if (vk.framebufImages[a]) vkDestroyImage(vk.device, vk.framebufImages[a], nullptr);
            if (vk.framebufMems[a]) vkFreeMemory(vk.device, vk.framebufMems[a], nullptr);
        }
        if (vk.cmdPool)         vkDestroyCommandPool(vk.device, vk.cmdPool, nullptr);
        vkDestroyDevice(vk.device, nullptr);
    }
    if (vk.instance) vkDestroyInstance(vk.instance, nullptr);
}

static uint32_t findMemoryType(VkPhysicalDevice pd, uint32_t typeBits,
                                VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(pd, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    return ~0u;
}

// Create a shader module from SPIR-V binary data.
static VkShaderModule createShaderModule(VkDevice dev,
                                         const std::vector<uint32_t> &spv) {
    VkShaderModuleCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = spv.size() * sizeof(uint32_t);
    ci.pCode = spv.data();
    VkShaderModule sm;
    vkCreateShaderModule(dev, &ci, nullptr, &sm);
    return sm;
}

// Create a 1x1 image + view + memory for a dummy texture.
struct DummyTexture {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

static DummyTexture createDummyTexture(VulkanState &vk, VkFormat fmt) {
    DummyTexture dt;
    VkImageCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent = {1, 1, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateImage(vk.device, &ici, nullptr, &dt.image);

    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(vk.device, dt.image, &mr);
    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = findMemoryType(vk.physicalDevice, mr.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(vk.device, &mai, nullptr, &dt.memory);
    vkBindImageMemory(vk.device, dt.image, dt.memory, 0);

    VkImageViewCreateInfo ivci = {};
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image = dt.image;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = fmt;
    ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.levelCount = 1;
    ivci.subresourceRange.layerCount = 1;
    vkCreateImageView(vk.device, &ivci, nullptr, &dt.view);
    return dt;
}

// Upload pixel data to a dummy texture via a staging buffer.
static void uploadDummyTexture(VulkanState &vk, DummyTexture &dt,
                                const uint8_t *rgba) {
    VkBuffer staging;
    VkDeviceMemory stagingMem;
    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = 4;  // 1 pixel RGBA8
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vkCreateBuffer(vk.device, &bci, nullptr, &staging);

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(vk.device, staging, &mr);
    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = findMemoryType(vk.physicalDevice, mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(vk.device, &mai, nullptr, &stagingMem);
    vkBindBufferMemory(vk.device, staging, stagingMem, 0);

    void *data;
    vkMapMemory(vk.device, stagingMem, 0, 4, 0, &data);
    memcpy(data, rgba, 4);
    vkUnmapMemory(vk.device, stagingMem);

    // Transition image layout and copy
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cbai = {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = vk.cmdPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    vkAllocateCommandBuffers(vk.device, &cbai, &cb);

    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &cbbi);

    VkImageMemoryBarrier bar = {};
    bar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.image = dt.image;
    bar.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bar.subresourceRange.levelCount = 1;
    bar.subresourceRange.layerCount = 1;
    bar.srcAccessMask = 0;
    bar.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &bar);

    VkBufferImageCopy region = {};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {1, 1, 1};
    vkCmdCopyBufferToImage(cb, staging, dt.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    bar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &bar);

    vkEndCommandBuffer(cb);

    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    vkQueueSubmit(vk.queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(vk.queue);

    vkFreeCommandBuffers(vk.device, vk.cmdPool, 1, &cb);
    vkDestroyBuffer(vk.device, staging, nullptr);
    vkFreeMemory(vk.device, stagingMem, nullptr);
}

static void destroyDummyTexture(VulkanState &vk, DummyTexture &dt) {
    if (dt.view)  vkDestroyImageView(vk.device, dt.view, nullptr);
    if (dt.image) vkDestroyImage(vk.device, dt.image, nullptr);
    if (dt.memory) vkFreeMemory(vk.device, dt.memory, nullptr);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <vertex.spv> <fragment.spv> [entry_point]\n",
                argv[0]);
        return 1;
    }

    const char *vertPath = argv[1];
    const char *fragPath = argv[2];

    // Load SPIR-V binaries
    auto vertSpv = readSpvFile(vertPath);
    auto fragSpv = readSpvFile(fragPath);
    if (vertSpv.empty() || fragSpv.empty()) return 1;

    // Reflect fragment SPIR-V for entry point and descriptor bindings
    SpvModule fragInfo;
    if (!reflectSpv(fragSpv, fragInfo)) return 1;

    // Override entry point name from CLI if provided
    std::string entryName = (argc >= 4) ? argv[3] : fragInfo.entryName;
    printf("Entry point: %s\n", entryName.c_str());
    printf("Texture bindings: %zu\n", fragInfo.texBindings.size());
    printf("Sampler bindings: %zu\n", fragInfo.sampBindings.size());

    VulkanState vk = {};

    // --- Instance ---
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &appInfo;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &vk.instance));

    // --- Physical device ---
    uint32_t pdCount = 0;
    vkEnumeratePhysicalDevices(vk.instance, &pdCount, nullptr);
    std::vector<VkPhysicalDevice> pds(pdCount);
    vkEnumeratePhysicalDevices(vk.instance, &pdCount, pds.data());

    // Pick the first discrete GPU, or fall back to first device.
    vk.physicalDevice = pds[0];
    for (auto pd : pds) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);
        printf("  GPU %u: %s (type=%d)\n", props.deviceID,
               props.deviceName, props.deviceType);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            vk.physicalDevice = pd;
    }
    VkPhysicalDeviceProperties pdProps;
    vkGetPhysicalDeviceProperties(vk.physicalDevice, &pdProps);
    printf("Using: %s\n", pdProps.deviceName);

    // --- Queue family ---
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk.physicalDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfProps(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(vk.physicalDevice, &qfCount, qfProps.data());
    vk.queueFamily = ~0u;
    for (uint32_t i = 0; i < qfCount; ++i) {
        if (qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            vk.queueFamily = i;
            break;
        }
    }
    if (vk.queueFamily == ~0u) {
        fprintf(stderr, "ERROR: no graphics queue\n");
        destroyVulkan(vk);
        return 1;
    }

    // --- Device ---
    float queuePri = 1.0f;
    VkDeviceQueueCreateInfo dqci = {};
    dqci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    dqci.queueFamilyIndex = vk.queueFamily;
    dqci.queueCount = 1;
    dqci.pQueuePriorities = &queuePri;

    VkDeviceCreateInfo dci = {};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &dqci;
    VK_CHECK(vkCreateDevice(vk.physicalDevice, &dci, nullptr, &vk.device));
    vkGetDeviceQueue(vk.device, vk.queueFamily, 0, &vk.queue);

    // --- Command pool ---
    VkCommandPoolCreateInfo cpci = {};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = vk.queueFamily;
    VK_CHECK(vkCreateCommandPool(vk.device, &cpci, nullptr, &vk.cmdPool));

    // --- Render pass (MAX_ATTACHMENTS color attachments) ---
    VkAttachmentDescription atts[MAX_ATTACHMENTS];
    VkAttachmentReference attRefs[MAX_ATTACHMENTS];
    for (uint32_t a = 0; a < MAX_ATTACHMENTS; ++a) {
        atts[a] = {};
        atts[a].format = VK_FORMAT_R32_SFLOAT;
        atts[a].samples = VK_SAMPLE_COUNT_1_BIT;
        atts[a].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        atts[a].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        atts[a].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        atts[a].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        attRefs[a] = {a, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    }

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = MAX_ATTACHMENTS;
    subpass.pColorAttachments = attRefs;

    VkRenderPassCreateInfo rpci = {};
    rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = MAX_ATTACHMENTS;
    rpci.pAttachments = atts;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &subpass;
    VK_CHECK(vkCreateRenderPass(vk.device, &rpci, nullptr, &vk.renderPass));

    // --- Framebuffer images (1x1, R32_SFLOAT, one per attachment) ---
    vk.attachmentCount = MAX_ATTACHMENTS;
    for (uint32_t a = 0; a < MAX_ATTACHMENTS; ++a) {
        VkImageCreateInfo fbici = {};
        fbici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        fbici.imageType = VK_IMAGE_TYPE_2D;
        fbici.format = VK_FORMAT_R32_SFLOAT;
        fbici.extent = {1, 1, 1};
        fbici.mipLevels = 1;
        fbici.arrayLayers = 1;
        fbici.samples = VK_SAMPLE_COUNT_1_BIT;
        fbici.tiling = VK_IMAGE_TILING_OPTIMAL;
        fbici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        fbici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VK_CHECK(vkCreateImage(vk.device, &fbici, nullptr, &vk.framebufImages[a]));

        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(vk.device, vk.framebufImages[a], &mr);
        VkMemoryAllocateInfo mai = {};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = findMemoryType(vk.physicalDevice, mr.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(vk.device, &mai, nullptr, &vk.framebufMems[a]));
        VK_CHECK(vkBindImageMemory(vk.device, vk.framebufImages[a], vk.framebufMems[a], 0));

        VkImageViewCreateInfo ivci = {};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = vk.framebufImages[a];
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = VK_FORMAT_R32_SFLOAT;
        ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ivci.subresourceRange.levelCount = 1;
        ivci.subresourceRange.layerCount = 1;
        VK_CHECK(vkCreateImageView(vk.device, &ivci, nullptr, &vk.framebufViews[a]));
    }

    {
        VkFramebufferCreateInfo fci = {};
        fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fci.renderPass = vk.renderPass;
        fci.attachmentCount = MAX_ATTACHMENTS;
        fci.pAttachments = vk.framebufViews;
        fci.width = 1;
        fci.height = 1;
        fci.layers = 1;
        VK_CHECK(vkCreateFramebuffer(vk.device, &fci, nullptr, &vk.framebuffer));
    }

    // --- Descriptor set layout ---
    // The SPIR-V uses separate OpTypeImage + OpTypeSampler (combined at
    // runtime with OpSampledImage), so we use SAMPLED_IMAGE for images
    // and SAMPLER for samplers — NOT COMBINED_IMAGE_SAMPLER.
    uint32_t bindingCount = static_cast<uint32_t>(
        fragInfo.texBindings.size() + fragInfo.sampBindings.size());
    std::vector<VkDescriptorSetLayoutBinding> layoutBindings(bindingCount);

    for (size_t bi = 0; bi < fragInfo.texBindings.size(); ++bi) {
        auto &lb = layoutBindings[bi];
        lb.binding = fragInfo.texBindings[bi];
        lb.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        lb.descriptorCount = 1;
        lb.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    for (size_t si = 0; si < fragInfo.sampBindings.size(); ++si) {
        auto &lb = layoutBindings[fragInfo.texBindings.size() + si];
        lb.binding = fragInfo.sampBindings[si];
        lb.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        lb.descriptorCount = 1;
        lb.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dslci = {};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = static_cast<uint32_t>(layoutBindings.size());
    dslci.pBindings = layoutBindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(vk.device, &dslci, nullptr, &vk.dsLayout));

    // --- Push constant range ---
    VkPushConstantRange pcRange = {};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcRange.offset = 0;
    pcRange.size = 8;  // two floats: ndotl, emissive

    // --- Pipeline layout ---
    VkPipelineLayoutCreateInfo plci = {};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &vk.dsLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcRange;
    VK_CHECK(vkCreatePipelineLayout(vk.device, &plci, nullptr, &vk.pipelineLayout));

    // --- Shader modules ---
    VkShaderModule vertMod = createShaderModule(vk.device, vertSpv);
    VkShaderModule fragMod = createShaderModule(vk.device, fragSpv);

    VkPipelineShaderStageCreateInfo vertStage = {};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertMod;
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage = {};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragMod;
    fragStage.pName = entryName.c_str();

    VkPipelineShaderStageCreateInfo stages[] = {vertStage, fragStage};

    // --- Vertex input (no vertex buffer — gl_VertexIndex) ---
    VkPipelineVertexInputStateCreateInfo visci = {};
    visci.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo iasci = {};
    iasci.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    iasci.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport = {0, 0, 1, 1, 0, 1};
    VkRect2D scissor = {{0, 0}, {1, 1}};
    VkPipelineViewportStateCreateInfo vpsci = {};
    vpsci.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpsci.viewportCount = 1;
    vpsci.pViewports = &viewport;
    vpsci.scissorCount = 1;
    vpsci.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rsci = {};
    rsci.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rsci.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo msci = {};
    msci.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msci.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cba[MAX_ATTACHMENTS];
    for (uint32_t a = 0; a < MAX_ATTACHMENTS; ++a) {
        cba[a] = {};
        cba[a].colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
    }

    VkPipelineColorBlendStateCreateInfo cbsci = {};
    cbsci.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cbsci.attachmentCount = MAX_ATTACHMENTS;
    cbsci.pAttachments = cba;

    VkGraphicsPipelineCreateInfo gpci = {};
    gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpci.stageCount = 2;
    gpci.pStages = stages;
    gpci.pVertexInputState = &visci;
    gpci.pInputAssemblyState = &iasci;
    gpci.pViewportState = &vpsci;
    gpci.pRasterizationState = &rsci;
    gpci.pMultisampleState = &msci;
    gpci.pColorBlendState = &cbsci;
    gpci.layout = vk.pipelineLayout;
    gpci.renderPass = vk.renderPass;
    gpci.subpass = 0;
    VK_CHECK(vkCreateGraphicsPipelines(vk.device, VK_NULL_HANDLE, 1, &gpci,
                                        nullptr, &vk.pipeline));

    vkDestroyShaderModule(vk.device, vertMod, nullptr);
    vkDestroyShaderModule(vk.device, fragMod, nullptr);

    // --- Descriptor pool and set ---
    std::vector<VkDescriptorPoolSize> poolSizes;
    if (!fragInfo.texBindings.empty())
        poolSizes.push_back({VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                             static_cast<uint32_t>(fragInfo.texBindings.size())});
    if (!fragInfo.sampBindings.empty())
        poolSizes.push_back({VK_DESCRIPTOR_TYPE_SAMPLER,
                             static_cast<uint32_t>(fragInfo.sampBindings.size())});

    VkDescriptorPoolCreateInfo dpci = {};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    dpci.pPoolSizes = poolSizes.data();
    VK_CHECK(vkCreateDescriptorPool(vk.device, &dpci, nullptr, &vk.dsPool));

    VkDescriptorSetAllocateInfo dsai = {};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = vk.dsPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &vk.dsLayout;
    VK_CHECK(vkAllocateDescriptorSets(vk.device, &dsai, &vk.ds));

    // --- Create dummy textures and samplers ---
    std::vector<DummyTexture> dummies(fragInfo.texBindings.size());
    uint8_t white[4] = {255, 255, 255, 255};
    for (size_t i = 0; i < dummies.size(); ++i) {
        dummies[i] = createDummyTexture(vk, VK_FORMAT_R8G8B8A8_UNORM);
        uploadDummyTexture(vk, dummies[i], white);
    }

    VkSampler sampler = VK_NULL_HANDLE;
    {
        VkSamplerCreateInfo sci = {};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = VK_FILTER_NEAREST;
        sci.minFilter = VK_FILTER_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(vk.device, &sci, nullptr, &sampler));
    }

    // --- Write descriptor set ---
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorImageInfo> imageInfos(dummies.size());
    for (size_t i = 0; i < dummies.size(); ++i) {
        imageInfos[i].imageView = dummies[i].view;
        imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfos[i].sampler = VK_NULL_HANDLE;  // not used for SAMPLED_IMAGE

        VkWriteDescriptorSet w = {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = vk.ds;
        w.dstBinding = fragInfo.texBindings[i];
        w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w.descriptorCount = 1;
        w.pImageInfo = &imageInfos[i];
        writes.push_back(w);
    }

    // Sampler-only bindings
    std::vector<VkDescriptorImageInfo> sampInfos(fragInfo.sampBindings.size());
    for (size_t i = 0; i < fragInfo.sampBindings.size(); ++i) {
        sampInfos[i].sampler = sampler;
        VkWriteDescriptorSet w = {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = vk.ds;
        w.dstBinding = fragInfo.sampBindings[i];
        w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        w.descriptorCount = 1;
        w.pImageInfo = &sampInfos[i];
        writes.push_back(w);
    }

    vkUpdateDescriptorSets(vk.device,
                           static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);

    // --- Command buffer: draw + readback ---
    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo cbai2 = {};
    cbai2.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai2.commandPool = vk.cmdPool;
    cbai2.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai2.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(vk.device, &cbai2, &cb));

    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &cbbi));

    VkClearValue clearVals[MAX_ATTACHMENTS];
    for (uint32_t a = 0; a < MAX_ATTACHMENTS; ++a)
        clearVals[a].color.float32[0] = -999.0f;
    VkRenderPassBeginInfo rpbi = {};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = vk.renderPass;
    rpbi.framebuffer = vk.framebuffer;
    rpbi.renderArea = {{0, 0}, {1, 1}};
    rpbi.clearValueCount = MAX_ATTACHMENTS;
    rpbi.pClearValues = clearVals;
    vkCmdBeginRenderPass(cb, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, vk.pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            vk.pipelineLayout, 0, 1, &vk.ds, 0, nullptr);

    // Push constants: ndotl=0.5, emissive=1.0
    float pcData[2] = {0.5f, 1.0f};
    vkCmdPushConstants(cb, vk.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                       0, sizeof(pcData), pcData);

    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRenderPass(cb);

    // Read back framebuffer: transition layout, copy each attachment to buffer
    VkImageMemoryBarrier fbBars[MAX_ATTACHMENTS];
    for (uint32_t a = 0; a < MAX_ATTACHMENTS; ++a) {
        fbBars[a] = {};
        fbBars[a].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        fbBars[a].image = vk.framebufImages[a];
        fbBars[a].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        fbBars[a].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        fbBars[a].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        fbBars[a].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        fbBars[a].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        fbBars[a].subresourceRange.levelCount = 1;
        fbBars[a].subresourceRange.layerCount = 1;
    }
    vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
        0, nullptr, 0, nullptr, MAX_ATTACHMENTS, fbBars);

    // Staging buffers for readback (one per attachment)
    VkBuffer readBufs[MAX_ATTACHMENTS];
    VkDeviceMemory readMems[MAX_ATTACHMENTS];
    for (uint32_t a = 0; a < MAX_ATTACHMENTS; ++a) {
        VkBufferCreateInfo bci = {};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = 4;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        vkCreateBuffer(vk.device, &bci, nullptr, &readBufs[a]);

        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(vk.device, readBufs[a], &mr);
        VkMemoryAllocateInfo mai = {};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = findMemoryType(vk.physicalDevice, mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(vk.device, &mai, nullptr, &readMems[a]);
        vkBindBufferMemory(vk.device, readBufs[a], readMems[a], 0);

        VkBufferImageCopy fbCopy = {};
        fbCopy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        fbCopy.imageSubresource.layerCount = 1;
        fbCopy.imageExtent = {1, 1, 1};
        vkCmdCopyImageToBuffer(cb, vk.framebufImages[a],
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               readBufs[a], 1, &fbCopy);
    }

    VK_CHECK(vkEndCommandBuffer(cb));

    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(vk.queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(vk.queue));

    // --- Read results from all attachments ---
    printf("\n=== RESULTS ===\n");
    printf("GPU: %s\n", pdProps.deviceName);
    int found = 0;
    for (uint32_t a = 0; a < MAX_ATTACHMENTS; ++a) {
        float result = 0.0f;
        void *data;
        vkMapMemory(vk.device, readMems[a], 0, 4, 0, &data);
        memcpy(&result, data, 4);
        vkUnmapMemory(vk.device, readMems[a]);
        if (result > -900.0f) {
            printf("  Location %u: %f\n", a, result);
            ++found;
        }
    }
    if (found == 0)
        printf("  (all outputs are clear value — shader wrote nothing)\n");

    // --- Cleanup ---
    for (uint32_t a = 0; a < MAX_ATTACHMENTS; ++a) {
        vkDestroyBuffer(vk.device, readBufs[a], nullptr);
        vkFreeMemory(vk.device, readMems[a], nullptr);
    }
    for (auto &dt : dummies) destroyDummyTexture(vk, dt);
    if (sampler) vkDestroySampler(vk.device, sampler, nullptr);
    destroyVulkan(vk);

    return 0;
}
