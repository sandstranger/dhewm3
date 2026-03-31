/*
===========================================================================

Doom 3 GPL Source Code
Copyright (C) 1999-2011 id Software LLC, a ZeniMax Media company.

This file is part of the Doom 3 GPL Source Code ("Doom 3 Source Code").

Doom 3 Source Code is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Doom 3 Source Code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Doom 3 Source Code.  If not, see <http://www.gnu.org/licenses/>.

In addition, the Doom 3 Source Code is also subject to certain additional terms. You should have received a copy of these additional terms immediately following the terms and conditions of the GNU General Public License which accompanied the Doom 3 Source Code.  If not, please request a copy in writing from id Software at the address below.

If you have questions concerning this license or the applicable additional terms, you may contact in writing id Software LLC, c/o ZeniMax Media Inc., Suite 120, Rockville, Maryland 20850 USA.

===========================================================================
*/

#ifdef ID_VULKAN

#include <vulkan/vulkan.h>

#include "sys/sys_sdl.h"

#if SDL_VERSION_ATLEAST(2, 0, 0)
#include <SDL_vulkan.h>
#endif

#include "renderer/RenderBackendPlatform.h"
#include "renderer/tr_local.h"
#include "renderer/vk/VulkanState.h"
#include "renderer/vk-ray/VulkanRTFuncs.h"
#include "framework/Common.h"
#include "framework/CVarSystem.h"
#include "sound/sound.h"
#include "sys/sys_imgui.h"
#include <cstring>
#include <vector>

// RT extension function pointers — loaded after vkCreateDevice when
// r_renderBackend is "vulkan-ray".  Defined here, declared extern in VulkanRTFuncs.h.
PFN_vkCreateAccelerationStructureKHR            pfn_vkCreateAccelerationStructureKHR;
PFN_vkDestroyAccelerationStructureKHR           pfn_vkDestroyAccelerationStructureKHR;
PFN_vkGetAccelerationStructureBuildSizesKHR     pfn_vkGetAccelerationStructureBuildSizesKHR;
PFN_vkCmdBuildAccelerationStructuresKHR         pfn_vkCmdBuildAccelerationStructuresKHR;
PFN_vkGetAccelerationStructureDeviceAddressKHR  pfn_vkGetAccelerationStructureDeviceAddressKHR;
PFN_vkCreateRayTracingPipelinesKHR              pfn_vkCreateRayTracingPipelinesKHR;
PFN_vkGetRayTracingShaderGroupHandlesKHR        pfn_vkGetRayTracingShaderGroupHandlesKHR;
PFN_vkCmdTraceRaysKHR                           pfn_vkCmdTraceRaysKHR;
PFN_vkGetBufferDeviceAddressKHR                 pfn_vkGetBufferDeviceAddressKHR;

// Global Vulkan state shared with draw backend and subsystems
vulkanState_t vkState;

// from glimp.cpp
extern bool			VkImp_Init( glimpParms_t parms );
extern void			VkImp_Shutdown();
extern SDL_Window *	VkImp_GetWindow();

#if ANDROID
PFN_vkGetPhysicalDeviceProperties2 p_vkGetPhysicalDeviceProperties2 = nullptr;

static void InitVulkanDeviceFunctions(VkInstance instance)
{
	p_vkGetPhysicalDeviceProperties2 =
			(PFN_vkGetPhysicalDeviceProperties2)
					vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2");
}
#endif


// ============================================================================
// Vulkan debug callback
// ============================================================================

#ifndef NDEBUG
static VKAPI_ATTR VkBool32 VKAPI_CALL VulkanDebugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
	void *pUserData )
{
	const char *prefix = "";
	if ( messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ) {
		prefix = "ERROR";
	} else if ( messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT ) {
		prefix = "WARNING";
	} else if ( messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT ) {
		prefix = "INFO";
	} else {
		prefix = "VERBOSE";
	}
	common->Printf( "Vulkan [%s]: %s\n", prefix, pCallbackData->pMessage );
	return VK_FALSE;
}
#endif

// ============================================================================
// Noop GPU command context (Vulkan doesn't use the fixed-function context)
// ============================================================================

static class idVulkanNoopGpuContext : public idRenderGpuCommandContext {
public:
	virtual void BindPipeline( idRenderGpuPipeline * ) {}
	virtual void SetBlendEnabled( bool ) {}
	virtual void SetScissorEnabled( bool ) {}
	virtual void SetTexture2DEnabled( bool ) {}
	virtual void SetBlendEquationAdd() {}
	virtual void Finish() {}
} s_vulkanNoopContext;

// ============================================================================
// idVulkanRenderBackendPlatform
// ============================================================================

class idVulkanRenderBackendPlatform : public idRenderBackendPlatform {
public:
	idVulkanRenderBackendPlatform();

	virtual renderBackendModule_t GetModule() const { return RBM_VULKAN; }

	virtual bool Init( const renderBackendConfig_t &config );
	virtual bool SetScreenParms( const renderBackendConfig_t &config );
	virtual void Shutdown();

	virtual void SwapBuffers();

	virtual bool SetSwapInterval( int swapInterval );
	virtual int GetSwapInterval() const;
	virtual float GetDisplayRefresh() const;
	virtual bool SetWindowResizable( bool enableResizable );
	virtual void ResetGamma();
	virtual void GetState( renderBackendState_t &state ) const;
	virtual gpuExtensionPointer_t GetExtensionPointer( const char *name ) const;
	virtual idRenderGpuCommandContext* GetImmediateContext();

	// Public accessors for other Vulkan code
	VkInstance		GetInstance() const { return instance; }
	VkPhysicalDevice GetPhysicalDevice() const { return physicalDevice; }
	VkDevice		GetDevice() const { return device; }
	VkQueue			GetGraphicsQueue() const { return graphicsQueue; }
	VkQueue			GetPresentQueue() const { return presentQueue; }
	uint32_t		GetGraphicsQueueFamily() const { return graphicsQueueFamily; }
	uint32_t		GetPresentQueueFamily() const { return presentQueueFamily; }
	VkSwapchainKHR	GetSwapchain() const { return swapchain; }
	VkFormat		GetSwapchainFormat() const { return swapchainFormat; }
	VkExtent2D		GetSwapchainExtent() const { return swapchainExtent; }
	uint32_t		GetSwapchainImageCount() const { return (uint32_t)swapchainImages.size(); }
	VkImageView		GetSwapchainImageView( uint32_t index ) const { return swapchainImageViews[index]; }
	VkFormat		GetDepthFormat() const { return depthFormat; }

private:
	bool			CreateInstance();
	bool			CreateSurface();
	bool			SelectPhysicalDevice();
	bool			CreateLogicalDevice();
	bool			CreateSwapchain( int width, int height );
	void			DestroySwapchain();
	VkFormat		FindDepthFormat();
	VkPresentModeKHR ChoosePresentMode( bool vsync );

	VkInstance					instance;
	VkSurfaceKHR				surface;
	VkPhysicalDevice			physicalDevice;
	VkDevice					device;
	VkQueue						graphicsQueue;
	VkQueue						presentQueue;
	uint32_t					graphicsQueueFamily;
	uint32_t					presentQueueFamily;
	VkSwapchainKHR				swapchain;
	VkFormat					swapchainFormat;
	VkExtent2D					swapchainExtent;
	std::vector<VkImage>		swapchainImages;
	std::vector<VkImageView>	swapchainImageViews;
	VkFormat					depthFormat;
	int							currentSwapInterval;
	bool						initialized;

#ifndef NDEBUG
	VkDebugUtilsMessengerEXT	debugMessenger;
#endif
};

// ---------------------------------------------------------------------------

idVulkanRenderBackendPlatform::idVulkanRenderBackendPlatform()
	: instance( VK_NULL_HANDLE )
	, surface( VK_NULL_HANDLE )
	, physicalDevice( VK_NULL_HANDLE )
	, device( VK_NULL_HANDLE )
	, graphicsQueue( VK_NULL_HANDLE )
	, presentQueue( VK_NULL_HANDLE )
	, graphicsQueueFamily( 0 )
	, presentQueueFamily( 0 )
	, swapchain( VK_NULL_HANDLE )
	, swapchainFormat( VK_FORMAT_B8G8R8A8_UNORM )
	, depthFormat( VK_FORMAT_D24_UNORM_S8_UINT )
	, currentSwapInterval( 1 )
	, initialized( false )
#ifndef NDEBUG
	, debugMessenger( VK_NULL_HANDLE )
#endif
{
	swapchainExtent.width = 0;
	swapchainExtent.height = 0;
}

// ---------------------------------------------------------------------------

bool idVulkanRenderBackendPlatform::CreateInstance() {
	VkApplicationInfo appInfo = {};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "dhewm3";
	appInfo.applicationVersion = VK_MAKE_VERSION( 1, 5, 5 );
	appInfo.pEngineName = "idTech 4";
	appInfo.engineVersion = VK_MAKE_VERSION( 4, 0, 0 );
	appInfo.apiVersion = VK_API_VERSION_1_3;

	// Get required extensions from SDL
	std::vector<const char *> extensions;

#if SDL_VERSION_ATLEAST(3, 0, 0)
	Uint32 sdlExtCount = 0;
	const char *const *sdlExts = SDL_Vulkan_GetInstanceExtensions( &sdlExtCount );
	if ( sdlExts == NULL ) {
		common->Warning( "SDL_Vulkan_GetInstanceExtensions failed: %s", SDL_GetError() );
		return false;
	}
	for ( Uint32 i = 0; i < sdlExtCount; i++ ) {
		extensions.push_back( sdlExts[i] );
	}
#elif SDL_VERSION_ATLEAST(2, 0, 0)
	SDL_Window *win = VkImp_GetWindow();
	unsigned int sdlExtCount = 0;
	if ( !SDL_Vulkan_GetInstanceExtensions( win, &sdlExtCount, NULL ) ) {
		common->Warning( "SDL_Vulkan_GetInstanceExtensions failed: %s", SDL_GetError() );
		return false;
	}
	extensions.resize( sdlExtCount );
	if ( !SDL_Vulkan_GetInstanceExtensions( win, &sdlExtCount, extensions.data() ) ) {
		common->Warning( "SDL_Vulkan_GetInstanceExtensions (2) failed: %s", SDL_GetError() );
		return false;
	}
#endif

	// Validation layers in debug builds
	std::vector<const char *> layers;
#ifndef NDEBUG
	layers.push_back( "VK_LAYER_KHRONOS_validation" );
	extensions.push_back( VK_EXT_DEBUG_UTILS_EXTENSION_NAME );
#endif

	VkInstanceCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo = &appInfo;
	createInfo.enabledExtensionCount = (uint32_t)extensions.size();
	createInfo.ppEnabledExtensionNames = extensions.data();
	createInfo.enabledLayerCount = (uint32_t)layers.size();
	createInfo.ppEnabledLayerNames = layers.data();

	VkResult result = vkCreateInstance( &createInfo, NULL, &instance );
	if ( result != VK_SUCCESS ) {
		// Retry without validation layers
		if ( !layers.empty() ) {
			common->Warning( "Vulkan validation layers not available, retrying without" );
			createInfo.enabledLayerCount = 0;
			createInfo.ppEnabledLayerNames = NULL;
			// Also remove debug utils extension
			extensions.pop_back();
			createInfo.enabledExtensionCount = (uint32_t)extensions.size();
			createInfo.ppEnabledExtensionNames = extensions.data();
			result = vkCreateInstance( &createInfo, NULL, &instance );
		}
		if ( result != VK_SUCCESS ) {
			common->Warning( "vkCreateInstance failed: %d", (int)result );
			return false;
		}
	}

	common->Printf( "Vulkan instance created\n" );

	// Set up debug messenger
#ifndef NDEBUG
	{
		PFN_vkCreateDebugUtilsMessengerEXT createDebugMessenger =
			(PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr( instance, "vkCreateDebugUtilsMessengerEXT" );
		if ( createDebugMessenger != NULL ) {
			VkDebugUtilsMessengerCreateInfoEXT debugInfo = {};
			debugInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
			debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
				| VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
			debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
				| VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
				| VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
			debugInfo.pfnUserCallback = VulkanDebugCallback;
			createDebugMessenger( instance, &debugInfo, NULL, &debugMessenger );
		}
	}
#endif

	return true;
}

bool idVulkanRenderBackendPlatform::CreateSurface() {
	SDL_Window *win = VkImp_GetWindow();

#if SDL_VERSION_ATLEAST(3, 0, 0)
	if ( !SDL_Vulkan_CreateSurface( win, instance, NULL, &surface ) ) {
#else
	if ( !SDL_Vulkan_CreateSurface( win, instance, &surface ) ) {
#endif
		common->Warning( "SDL_Vulkan_CreateSurface failed: %s", SDL_GetError() );
		return false;
	}
	common->Printf( "Vulkan surface created\n" );
	return true;
}

bool idVulkanRenderBackendPlatform::SelectPhysicalDevice() {
	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices( instance, &deviceCount, NULL );
	if ( deviceCount == 0 ) {
		common->Warning( "No Vulkan physical devices found" );
		return false;
	}

	std::vector<VkPhysicalDevice> devices( deviceCount );
	vkEnumeratePhysicalDevices( instance, &deviceCount, devices.data() );

	// Prefer discrete GPUs
	VkPhysicalDevice bestDevice = VK_NULL_HANDLE;
	int bestScore = -1;

	for ( uint32_t i = 0; i < deviceCount; i++ ) {
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties( devices[i], &props );

		// Check for required queue families
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties( devices[i], &queueFamilyCount, NULL );
		std::vector<VkQueueFamilyProperties> queueFamilies( queueFamilyCount );
		vkGetPhysicalDeviceQueueFamilyProperties( devices[i], &queueFamilyCount, queueFamilies.data() );

		bool hasGraphics = false;
		bool hasPresent = false;
		for ( uint32_t q = 0; q < queueFamilyCount; q++ ) {
			if ( queueFamilies[q].queueFlags & VK_QUEUE_GRAPHICS_BIT ) {
				hasGraphics = true;
			}
			VkBool32 presentSupport = VK_FALSE;
			vkGetPhysicalDeviceSurfaceSupportKHR( devices[i], q, surface, &presentSupport );
			if ( presentSupport ) {
				hasPresent = true;
			}
		}

		if ( !hasGraphics || !hasPresent ) {
			continue;
		}

		// Check extension support
		uint32_t extCount = 0;
		vkEnumerateDeviceExtensionProperties( devices[i], NULL, &extCount, NULL );
		std::vector<VkExtensionProperties> exts( extCount );
		vkEnumerateDeviceExtensionProperties( devices[i], NULL, &extCount, exts.data() );

		bool hasSwapchain = false;
		bool hasRT        = false;
		for ( uint32_t e = 0; e < extCount; e++ ) {
			if ( strcmp( exts[e].extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME ) == 0 ) {
				hasSwapchain = true;
			}
			if ( strcmp( exts[e].extensionName, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME ) == 0 ) {
				hasRT = true;
			}
		}
		if ( !hasSwapchain ) {
			continue;
		}

		bool needRT = idStr::Icmp( r_renderBackend.GetString(), "vulkan-ray" ) == 0;
		if ( needRT && !hasRT ) {
			common->Printf( "  Vulkan device %d: %s — no ray tracing support, skipping\n",
			                i, props.deviceName );
			continue;
		}

		int score = 0;
		if ( props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ) {
			score = 1000;
		} else if ( props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ) {
			score = 100;
		}

		common->Printf( "  Vulkan device %d: %s (score %d%s)\n",
		                i, props.deviceName, score, hasRT ? ", RT" : "" );

		if ( score > bestScore ) {
			bestScore = score;
			bestDevice = devices[i];
		}
	}

	if ( bestDevice == VK_NULL_HANDLE ) {
		bool needRT = idStr::Icmp( r_renderBackend.GetString(), "vulkan-ray" ) == 0;
		if ( needRT ) {
			common->Warning( "No Vulkan device with ray tracing support found. "
			                 "Use r_renderBackend vulkan for the rasterization backend." );
		} else {
			common->Warning( "No suitable Vulkan device found" );
		}
		return false;
	}

	physicalDevice = bestDevice;

	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties( physicalDevice, &props );
	common->Printf( "Selected Vulkan device: %s\n", props.deviceName );

	return true;
}

bool idVulkanRenderBackendPlatform::CreateLogicalDevice() {
	// Find graphics and present queue families
	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties( physicalDevice, &queueFamilyCount, NULL );
	std::vector<VkQueueFamilyProperties> queueFamilies( queueFamilyCount );
	vkGetPhysicalDeviceQueueFamilyProperties( physicalDevice, &queueFamilyCount, queueFamilies.data() );

	graphicsQueueFamily = UINT32_MAX;
	presentQueueFamily = UINT32_MAX;

	for ( uint32_t i = 0; i < queueFamilyCount; i++ ) {
		if ( queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT ) {
			graphicsQueueFamily = i;
		}
		VkBool32 presentSupport = VK_FALSE;
		vkGetPhysicalDeviceSurfaceSupportKHR( physicalDevice, i, surface, &presentSupport );
		if ( presentSupport ) {
			presentQueueFamily = i;
		}
		// Prefer a queue family that supports both
		if ( ( queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT ) && presentSupport ) {
			graphicsQueueFamily = i;
			presentQueueFamily = i;
			break;
		}
	}

	if ( graphicsQueueFamily == UINT32_MAX || presentQueueFamily == UINT32_MAX ) {
		common->Warning( "Vulkan: No suitable queue families" );
		return false;
	}

	// Create device with one or two queues
	float queuePriority = 1.0f;
	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

	VkDeviceQueueCreateInfo qci = {};
	qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	qci.queueFamilyIndex = graphicsQueueFamily;
	qci.queueCount = 1;
	qci.pQueuePriorities = &queuePriority;
	queueCreateInfos.push_back( qci );

	if ( presentQueueFamily != graphicsQueueFamily ) {
		VkDeviceQueueCreateInfo qci2 = {};
		qci2.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		qci2.queueFamilyIndex = presentQueueFamily;
		qci2.queueCount = 1;
		qci2.pQueuePriorities = &queuePriority;
		queueCreateInfos.push_back( qci2 );
	}

	bool useRT = idStr::Icmp( r_renderBackend.GetString(), "vulkan-ray" ) == 0;

	std::vector<const char *> devExts = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
	if ( useRT ) {
		devExts.push_back( VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME );
		devExts.push_back( VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME );
		devExts.push_back( VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME );
		devExts.push_back( VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME );
		devExts.push_back( VK_KHR_SPIRV_1_4_EXTENSION_NAME );
		devExts.push_back( VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME );
	}

	// Build feature chain: use VkPhysicalDeviceFeatures2 + pNext when RT
	// is enabled so we can enable RT-specific feature structs.
	VkPhysicalDeviceBufferDeviceAddressFeaturesKHR bdaFeatures = {};
	bdaFeatures.sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR;
	bdaFeatures.bufferDeviceAddress = VK_TRUE;

	VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures = {};
	asFeatures.sType                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
	asFeatures.accelerationStructure = VK_TRUE;
	asFeatures.pNext                 = &bdaFeatures;

	VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeatures = {};
	rtFeatures.sType              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
	rtFeatures.rayTracingPipeline = VK_TRUE;
	rtFeatures.pNext              = &asFeatures;

	VkPhysicalDeviceFeatures2 features2 = {};
	features2.sType                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	features2.features.samplerAnisotropy = VK_TRUE;
	features2.pNext                      = useRT ? (void *)&rtFeatures : NULL;

	VkDeviceCreateInfo deviceCreateInfo = {};
	deviceCreateInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceCreateInfo.pNext                   = &features2;   // use features2 chain
	deviceCreateInfo.pEnabledFeatures        = NULL;         // must be NULL when pNext has features2
	deviceCreateInfo.queueCreateInfoCount    = (uint32_t)queueCreateInfos.size();
	deviceCreateInfo.pQueueCreateInfos       = queueCreateInfos.data();
	deviceCreateInfo.enabledExtensionCount   = (uint32_t)devExts.size();
	deviceCreateInfo.ppEnabledExtensionNames = devExts.data();

	VkResult result = vkCreateDevice( physicalDevice, &deviceCreateInfo, NULL, &device );
	if ( result != VK_SUCCESS ) {
		common->Warning( "vkCreateDevice failed: %d", (int)result );
		return false;
	}

	vkGetDeviceQueue( device, graphicsQueueFamily, 0, &graphicsQueue );
	vkGetDeviceQueue( device, presentQueueFamily, 0, &presentQueue );

	if ( useRT ) {
		// Load RT extension function pointers
#define LOAD_PFN( name ) \
		pfn_##name = (PFN_##name)vkGetDeviceProcAddr( device, #name ); \
		if ( !pfn_##name ) { common->Warning( "RT: failed to load " #name ); }
		LOAD_PFN( vkCreateAccelerationStructureKHR )
		LOAD_PFN( vkDestroyAccelerationStructureKHR )
		LOAD_PFN( vkGetAccelerationStructureBuildSizesKHR )
		LOAD_PFN( vkCmdBuildAccelerationStructuresKHR )
		LOAD_PFN( vkGetAccelerationStructureDeviceAddressKHR )
		LOAD_PFN( vkCreateRayTracingPipelinesKHR )
		LOAD_PFN( vkGetRayTracingShaderGroupHandlesKHR )
		LOAD_PFN( vkCmdTraceRaysKHR )
		LOAD_PFN( vkGetBufferDeviceAddressKHR )
#undef LOAD_PFN
		common->Printf( "Vulkan ray tracing extension function pointers loaded\n" );
	}

	common->Printf( "Vulkan logical device created (graphics queue family %d, present queue family %d)\n",
		graphicsQueueFamily, presentQueueFamily );

	return true;
}

VkFormat idVulkanRenderBackendPlatform::FindDepthFormat() {
	VkFormat candidates[] = {
		VK_FORMAT_D32_SFLOAT_S8_UINT,
		VK_FORMAT_D24_UNORM_S8_UINT,
		VK_FORMAT_D32_SFLOAT
	};

	for ( int i = 0; i < 3; i++ ) {
		VkFormatProperties props;
		vkGetPhysicalDeviceFormatProperties( physicalDevice, candidates[i], &props );
		if ( props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT ) {
			return candidates[i];
		}
	}

	return VK_FORMAT_D24_UNORM_S8_UINT; // fallback
}

VkPresentModeKHR idVulkanRenderBackendPlatform::ChoosePresentMode( bool vsync ) {
	if ( vsync ) {
		return VK_PRESENT_MODE_FIFO_KHR; // always available, vsync
	}

	// Try mailbox (triple-buffer, no tearing, low latency)
	uint32_t modeCount = 0;
	vkGetPhysicalDeviceSurfacePresentModesKHR( physicalDevice, surface, &modeCount, NULL );
	std::vector<VkPresentModeKHR> modes( modeCount );
	vkGetPhysicalDeviceSurfacePresentModesKHR( physicalDevice, surface, &modeCount, modes.data() );

	for ( uint32_t i = 0; i < modeCount; i++ ) {
		if ( modes[i] == VK_PRESENT_MODE_MAILBOX_KHR ) {
			return VK_PRESENT_MODE_MAILBOX_KHR;
		}
	}

	// Fall back to immediate (no vsync, may tear)
	for ( uint32_t i = 0; i < modeCount; i++ ) {
		if ( modes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR ) {
			return VK_PRESENT_MODE_IMMEDIATE_KHR;
		}
	}

	return VK_PRESENT_MODE_FIFO_KHR;
}

bool idVulkanRenderBackendPlatform::CreateSwapchain( int width, int height ) {
	VkSurfaceCapabilitiesKHR caps;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR( physicalDevice, surface, &caps );

	// Choose extent
	if ( caps.currentExtent.width != UINT32_MAX ) {
		swapchainExtent = caps.currentExtent;
	} else {
		swapchainExtent.width = (uint32_t)width;
		swapchainExtent.height = (uint32_t)height;
		if ( swapchainExtent.width < caps.minImageExtent.width ) swapchainExtent.width = caps.minImageExtent.width;
		if ( swapchainExtent.width > caps.maxImageExtent.width ) swapchainExtent.width = caps.maxImageExtent.width;
		if ( swapchainExtent.height < caps.minImageExtent.height ) swapchainExtent.height = caps.minImageExtent.height;
		if ( swapchainExtent.height > caps.maxImageExtent.height ) swapchainExtent.height = caps.maxImageExtent.height;
	}

	// Choose surface format (prefer B8G8R8A8_SRGB or B8G8R8A8_UNORM)
	uint32_t formatCount = 0;
	vkGetPhysicalDeviceSurfaceFormatsKHR( physicalDevice, surface, &formatCount, NULL );
	std::vector<VkSurfaceFormatKHR> formats( formatCount );
	vkGetPhysicalDeviceSurfaceFormatsKHR( physicalDevice, surface, &formatCount, formats.data() );

	swapchainFormat = formats[0].format;
	VkColorSpaceKHR colorSpace = formats[0].colorSpace;

	for ( uint32_t i = 0; i < formatCount; i++ ) {
		if ( formats[i].format == VK_FORMAT_B8G8R8A8_SRGB
			&& formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR ) {
			swapchainFormat = formats[i].format;
			colorSpace = formats[i].colorSpace;
			break;
		}
		if ( formats[i].format == VK_FORMAT_B8G8R8A8_UNORM ) {
			swapchainFormat = formats[i].format;
			colorSpace = formats[i].colorSpace;
		}
	}

	// Image count (double-buffer minimum, prefer triple)
	uint32_t imageCount = caps.minImageCount + 1;
	if ( caps.maxImageCount > 0 && imageCount > caps.maxImageCount ) {
		imageCount = caps.maxImageCount;
	}

	VkPresentModeKHR presentMode = ChoosePresentMode( currentSwapInterval > 0 );

	VkSwapchainCreateInfoKHR swapchainInfo = {};
	swapchainInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	swapchainInfo.surface = surface;
	swapchainInfo.minImageCount = imageCount;
	swapchainInfo.imageFormat = swapchainFormat;
	swapchainInfo.imageColorSpace = colorSpace;
	swapchainInfo.imageExtent = swapchainExtent;
	swapchainInfo.imageArrayLayers = 1;
	swapchainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
		| VK_IMAGE_USAGE_TRANSFER_SRC_BIT;	// needed for CopyRender (mirrors, etc.)

	uint32_t queueFamilyIndices[] = { graphicsQueueFamily, presentQueueFamily };
	if ( graphicsQueueFamily != presentQueueFamily ) {
		swapchainInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		swapchainInfo.queueFamilyIndexCount = 2;
		swapchainInfo.pQueueFamilyIndices = queueFamilyIndices;
	} else {
		swapchainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	}

	swapchainInfo.preTransform = caps.currentTransform;
	swapchainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	swapchainInfo.presentMode = presentMode;
	swapchainInfo.clipped = VK_TRUE;
	swapchainInfo.oldSwapchain = VK_NULL_HANDLE;

	VkResult result = vkCreateSwapchainKHR( device, &swapchainInfo, NULL, &swapchain );
	if ( result != VK_SUCCESS ) {
		common->Warning( "vkCreateSwapchainKHR failed: %d", (int)result );
		return false;
	}

	// Get swapchain images
	vkGetSwapchainImagesKHR( device, swapchain, &imageCount, NULL );
	swapchainImages.resize( imageCount );
	vkGetSwapchainImagesKHR( device, swapchain, &imageCount, swapchainImages.data() );

	// Create image views
	swapchainImageViews.resize( imageCount );
	for ( uint32_t i = 0; i < imageCount; i++ ) {
		VkImageViewCreateInfo viewInfo = {};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = swapchainImages[i];
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = swapchainFormat;
		viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		result = vkCreateImageView( device, &viewInfo, NULL, &swapchainImageViews[i] );
		if ( result != VK_SUCCESS ) {
			common->Warning( "vkCreateImageView failed for swapchain image %d", i );
			return false;
		}
	}

	common->Printf( "Vulkan swapchain created: %dx%d, %d images, format %d, present mode %d\n",
		swapchainExtent.width, swapchainExtent.height, imageCount,
		(int)swapchainFormat, (int)presentMode );

	return true;
}

void idVulkanRenderBackendPlatform::DestroySwapchain() {
	if ( device == VK_NULL_HANDLE ) {
		return;
	}

	for ( size_t i = 0; i < swapchainImageViews.size(); i++ ) {
		if ( swapchainImageViews[i] != VK_NULL_HANDLE ) {
			vkDestroyImageView( device, swapchainImageViews[i], NULL );
		}
	}
	swapchainImageViews.clear();
	swapchainImages.clear();

	if ( swapchain != VK_NULL_HANDLE ) {
		vkDestroySwapchainKHR( device, swapchain, NULL );
		swapchain = VK_NULL_HANDLE;
	}
}

// ============================================================================
// idRenderBackendPlatform interface implementation
// ============================================================================

bool idVulkanRenderBackendPlatform::Init( const renderBackendConfig_t &config ) {
	common->Printf( "----- Initializing Vulkan -----\n" );

	// Create SDL window with Vulkan flag
	glimpParms_t parms;
	memset( &parms, 0, sizeof( parms ) );
	parms.width = config.width;
	parms.height = config.height;
	parms.fullScreen = config.fullScreen;
	parms.fullScreenDesktop = config.fullScreenDesktop;
	parms.stereo = config.stereo;
	parms.displayHz = config.displayHz;
	parms.multiSamples = config.multiSamples;

	if ( !VkImp_Init( parms ) ) {
		common->Warning( "Vulkan: Failed to create SDL window" );
		return false;
	}

	// Input and sound are tied to the window
	Sys_InitInput();
	soundSystem->InitHW();

	if ( !CreateInstance() ) {
		VkImp_Shutdown();
		return false;
	}

	if ( !CreateSurface() ) {
		Shutdown();
		return false;
	}

	if ( !SelectPhysicalDevice() ) {
		Shutdown();
		return false;
	}

	if ( !CreateLogicalDevice() ) {
		Shutdown();
		return false;
	}

	depthFormat = FindDepthFormat();

	if ( !CreateSwapchain( config.width, config.height ) ) {
		Shutdown();
		return false;
	}

	// Populate glConfig for the rest of the engine
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties( physicalDevice, &props );
	glConfig.vendor_string = "Vulkan";
	glConfig.renderer_string = props.deviceName;
	glConfig.version_string = "Vulkan 1.0";
	glConfig.extensions_string = "";
	glConfig.maxTextureSize = props.limits.maxImageDimension2D;
	glConfig.isInitialized = true;
	glConfig.isVulkan = true;

	// Report anisotropy support
	VkPhysicalDeviceFeatures features;
	vkGetPhysicalDeviceFeatures( physicalDevice, &features );
	glConfig.anisotropicAvailable = features.samplerAnisotropy ? true : false;
	glConfig.maxTextureAnisotropy = props.limits.maxSamplerAnisotropy;

	common->Printf( "Vulkan renderer: %s\n", props.deviceName );
	common->Printf( "Vulkan API version: %d.%d.%d\n",
		VK_VERSION_MAJOR( props.apiVersion ),
		VK_VERSION_MINOR( props.apiVersion ),
		VK_VERSION_PATCH( props.apiVersion ) );

	// Populate global Vulkan state for subsystems
	vkState.instance = instance;
	vkState.physicalDevice = physicalDevice;
	vkState.device = device;
	vkState.graphicsQueue = graphicsQueue;
	vkState.presentQueue = presentQueue;
	vkState.graphicsQueueFamily = graphicsQueueFamily;
	vkState.presentQueueFamily = presentQueueFamily;
	vkState.swapchain = swapchain;
	vkState.swapchainFormat = swapchainFormat;
	vkState.swapchainExtent = swapchainExtent;
	vkState.depthFormat = depthFormat;
	vkState.swapchainImages = swapchainImages;
	vkState.swapchainImageViews = swapchainImageViews;

	initialized = true;
#if ANDROID
	InitVulkanDeviceFunctions(instance);
#endif
	return true;
}

bool idVulkanRenderBackendPlatform::SetScreenParms( const renderBackendConfig_t &config ) {
	if ( !initialized ) {
		return false;
	}

	// Recreate swapchain with new size
	vkDeviceWaitIdle( device );
	DestroySwapchain();

	if ( !CreateSwapchain( config.width, config.height ) ) {
		return false;
	}

	// Keep global state in sync
	vkState.swapchain = swapchain;
	vkState.swapchainFormat = swapchainFormat;
	vkState.swapchainExtent = swapchainExtent;
	vkState.swapchainImages = swapchainImages;
	vkState.swapchainImageViews = swapchainImageViews;

	glConfig.vidWidth = swapchainExtent.width;
	glConfig.vidHeight = swapchainExtent.height;

	return true;
}

void idVulkanRenderBackendPlatform::Shutdown() {
	common->Printf( "Shutting down Vulkan\n" );

	if ( device != VK_NULL_HANDLE ) {
		vkDeviceWaitIdle( device );
	}

	// Shut down ImGui before destroying the Vulkan device — the ImGui
	// Vulkan backend has pipelines and descriptor pools that need the
	// device to still be alive for cleanup.
	D3::ImGuiHooks::Shutdown();

	DestroySwapchain();

	if ( device != VK_NULL_HANDLE ) {
		vkDestroyDevice( device, NULL );
		device = VK_NULL_HANDLE;
	}

	if ( surface != VK_NULL_HANDLE && instance != VK_NULL_HANDLE ) {
		vkDestroySurfaceKHR( instance, surface, NULL );
		surface = VK_NULL_HANDLE;
	}

#ifndef NDEBUG
	if ( debugMessenger != VK_NULL_HANDLE && instance != VK_NULL_HANDLE ) {
		PFN_vkDestroyDebugUtilsMessengerEXT destroyDebugMessenger =
			(PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr( instance, "vkDestroyDebugUtilsMessengerEXT" );
		if ( destroyDebugMessenger != NULL ) {
			destroyDebugMessenger( instance, debugMessenger, NULL );
		}
		debugMessenger = VK_NULL_HANDLE;
	}
#endif

	if ( instance != VK_NULL_HANDLE ) {
		vkDestroyInstance( instance, NULL );
		instance = VK_NULL_HANDLE;
	}

	VkImp_Shutdown();

	physicalDevice = VK_NULL_HANDLE;
	graphicsQueue = VK_NULL_HANDLE;
	presentQueue = VK_NULL_HANDLE;
	initialized = false;
	glConfig.isInitialized = false;
}

void idVulkanRenderBackendPlatform::SwapBuffers() {
	// Presentation happens through the Vulkan draw backend's command submission.
	// This method is kept for the platform interface contract.
}

bool idVulkanRenderBackendPlatform::SetSwapInterval( int swapInterval ) {
	if ( !initialized ) {
		return false;
	}

	if ( swapInterval == currentSwapInterval ) {
		return true;
	}

	currentSwapInterval = swapInterval;

	// Recreate swapchain with new present mode
	vkDeviceWaitIdle( device );
	DestroySwapchain();
	if ( !CreateSwapchain( swapchainExtent.width, swapchainExtent.height ) ) {
		return false;
	}

	// Keep global state in sync
	vkState.swapchain = swapchain;
	vkState.swapchainFormat = swapchainFormat;
	vkState.swapchainExtent = swapchainExtent;
	vkState.swapchainImages = swapchainImages;
	vkState.swapchainImageViews = swapchainImageViews;
	return true;
}

int idVulkanRenderBackendPlatform::GetSwapInterval() const {
	return currentSwapInterval;
}

float idVulkanRenderBackendPlatform::GetDisplayRefresh() const {
	return GLimp_GetDisplayRefresh(); // SDL handles this
}

bool idVulkanRenderBackendPlatform::SetWindowResizable( bool enableResizable ) {
	return GLimp_SetWindowResizable( enableResizable ); // SDL handles this
}

void idVulkanRenderBackendPlatform::ResetGamma() {
	// Vulkan doesn't use hardware gamma ramps in the same way
}

void idVulkanRenderBackendPlatform::GetState( renderBackendState_t &state ) const {
	memset( &state, 0, sizeof( state ) );
	if ( !initialized ) {
		return;
	}
	state.width = swapchainExtent.width;
	state.height = swapchainExtent.height;
	state.fullScreen = glConfig.isFullscreen;
	state.swapInterval = currentSwapInterval;
	state.displayRefreshHz = GLimp_GetDisplayRefresh();
}

gpuExtensionPointer_t idVulkanRenderBackendPlatform::GetExtensionPointer( const char * ) const {
	return NULL; // Vulkan doesn't use GL extension pointers
}

idRenderGpuCommandContext* idVulkanRenderBackendPlatform::GetImmediateContext() {
	return &s_vulkanNoopContext;
}

// ============================================================================
// Factory
// ============================================================================

static idVulkanRenderBackendPlatform s_vulkanPlatform;

idRenderBackendPlatform* R_GetVulkanBackendPlatform() {
	return &s_vulkanPlatform;
}

#endif // ID_VULKAN
