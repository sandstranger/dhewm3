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

#include "renderer/vk-ray/VulkanRayPipeline.h"
#include "renderer/vk-ray/VulkanRTFuncs.h"
#include "framework/Common.h"
#include <cstring>
#include <vector>
#include <functional>
#include <cstdint>

// Embedded SPIR-V (generated at build time from RT shaders)
#include "generated/rt_raygen_rgen_spv.h"
#include "generated/rt_miss_rmiss_spv.h"
#include "generated/rt_shadow_miss_rmiss_spv.h"
#include "generated/rt_hit_rchit_spv.h"

idVulkanRayPipelineManager vkRayPipelineMgr;

idVulkanRayPipelineManager::idVulkanRayPipelineManager()
	: device( VK_NULL_HANDLE )
	, physicalDevice( VK_NULL_HANDLE )
	, oneTimePool( VK_NULL_HANDLE )
	, graphicsQueue( VK_NULL_HANDLE )
	, outputImage( VK_NULL_HANDLE )
	, outputMemory( VK_NULL_HANDLE )
	, outputView( VK_NULL_HANDLE )
	, descSetLayout( VK_NULL_HANDLE )
	, descPool( VK_NULL_HANDLE )
	, descSet( VK_NULL_HANDLE )
	, pipelineLayout( VK_NULL_HANDLE )
	, rtPipeline( VK_NULL_HANDLE )
	, sbtBuffer( VK_NULL_HANDLE )
	, sbtMemory( VK_NULL_HANDLE )
	, initialized( false )
{
	memset( &raygenRegion,   0, sizeof(raygenRegion) );
	memset( &missRegion,     0, sizeof(missRegion) );
	memset( &hitRegion,      0, sizeof(hitRegion) );
	memset( &callableRegion, 0, sizeof(callableRegion) );
}

// ---------------------------------------------------------------------------

bool idVulkanRayPipelineManager::Init( VkDevice dev, VkPhysicalDevice phyDev,
                                       VkExtent2D ext, uint32_t queueFamily,
                                       VkQueue queue ) {
	device        = dev;
	physicalDevice = phyDev;
	extent        = ext;
	graphicsQueue = queue;

	VkCommandPoolCreateInfo poolCI = {};
	poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolCI.queueFamilyIndex = queueFamily;
	poolCI.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
	if ( vkCreateCommandPool( device, &poolCI, NULL, &oneTimePool ) != VK_SUCCESS ) {
		common->Warning( "RayPipeline: failed to create command pool" );
		return false;
	}

	if ( !CreateOutputImage( ext.width, ext.height ) ) return false;
	if ( !CreateDescriptorSetLayout() ) return false;
	if ( !CreateDescriptorPool() ) return false;
	if ( !CreateDescriptorSet() ) return false;
	if ( !CreatePipeline() ) return false;
	if ( !CreateSBT() ) return false;

	// Always allocate the benchmark SSBO (large in benchmark mode, stub otherwise).
	// This keeps the descriptor set binding 4 valid in all cases.
	if ( !CreateBenchmarkBuffer() ) {
		common->Warning( "RayPipeline: failed to create benchmark buffer" );
	}

	// Bind the output image view to descriptor set binding 1 (static)
	VkDescriptorImageInfo imgInfo = {};
	imgInfo.imageView   = outputView;
	imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkWriteDescriptorSet wds = {};
	wds.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	wds.dstSet          = descSet;
	wds.dstBinding      = 1;
	wds.descriptorCount = 1;
	wds.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	wds.pImageInfo      = &imgInfo;
	vkUpdateDescriptorSets( device, 1, &wds, 0, NULL );

	initialized = true;
	common->Printf( "Vulkan ray tracing pipeline initialized\n" );
	return true;
}

void idVulkanRayPipelineManager::Shutdown() {
	if ( !initialized ) return;

	vkDeviceWaitIdle( device );

	if ( benchmarkBuffer != VK_NULL_HANDLE ) {
		if ( benchmarkMapped ) {
			vkUnmapMemory( device, benchmarkMemory );
			benchmarkMapped = nullptr;
		}
		vkDestroyBuffer( device, benchmarkBuffer, NULL );  benchmarkBuffer = VK_NULL_HANDLE;
		vkFreeMemory( device, benchmarkMemory, NULL );     benchmarkMemory = VK_NULL_HANDLE;
	}

	if ( sbtBuffer   != VK_NULL_HANDLE ) { vkDestroyBuffer( device, sbtBuffer, NULL );   sbtBuffer  = VK_NULL_HANDLE; }
	if ( sbtMemory   != VK_NULL_HANDLE ) { vkFreeMemory( device, sbtMemory, NULL );       sbtMemory  = VK_NULL_HANDLE; }
	if ( rtPipeline  != VK_NULL_HANDLE ) { vkDestroyPipeline( device, rtPipeline, NULL ); rtPipeline = VK_NULL_HANDLE; }
	if ( pipelineLayout != VK_NULL_HANDLE ) { vkDestroyPipelineLayout( device, pipelineLayout, NULL ); pipelineLayout = VK_NULL_HANDLE; }
	if ( descPool    != VK_NULL_HANDLE ) { vkDestroyDescriptorPool( device, descPool, NULL );  descPool = VK_NULL_HANDLE; }
	if ( descSetLayout != VK_NULL_HANDLE ) { vkDestroyDescriptorSetLayout( device, descSetLayout, NULL ); descSetLayout = VK_NULL_HANDLE; }
	if ( outputView  != VK_NULL_HANDLE ) { vkDestroyImageView( device, outputView, NULL );  outputView  = VK_NULL_HANDLE; }
	if ( outputImage != VK_NULL_HANDLE ) { vkDestroyImage( device, outputImage, NULL );      outputImage = VK_NULL_HANDLE; }
	if ( outputMemory != VK_NULL_HANDLE ) { vkFreeMemory( device, outputMemory, NULL );      outputMemory = VK_NULL_HANDLE; }
	if ( oneTimePool != VK_NULL_HANDLE ) { vkDestroyCommandPool( device, oneTimePool, NULL ); oneTimePool = VK_NULL_HANDLE; }

	initialized = false;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

uint32_t idVulkanRayPipelineManager::FindMemoryType( uint32_t filter,
                                                      VkMemoryPropertyFlags props ) const {
	VkPhysicalDeviceMemoryProperties memProp;
	vkGetPhysicalDeviceMemoryProperties( physicalDevice, &memProp );
	for ( uint32_t i = 0; i < memProp.memoryTypeCount; i++ ) {
		if ( (filter & (1u << i)) && (memProp.memoryTypes[i].propertyFlags & props) == props ) {
			return i;
		}
	}
	return UINT32_MAX;
}

void idVulkanRayPipelineManager::SubmitOneTime( std::function<void(VkCommandBuffer)> fn ) {
	VkCommandBufferAllocateInfo ai = {};
	ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	ai.commandPool        = oneTimePool;
	ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	ai.commandBufferCount = 1;
	VkCommandBuffer cmd;
	if ( vkAllocateCommandBuffers( device, &ai, &cmd ) != VK_SUCCESS ) return;

	VkCommandBufferBeginInfo bi = {};
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer( cmd, &bi );
	fn( cmd );
	vkEndCommandBuffer( cmd );

	VkSubmitInfo si = {};
	si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.commandBufferCount = 1;
	si.pCommandBuffers    = &cmd;
	vkQueueSubmit( graphicsQueue, 1, &si, VK_NULL_HANDLE );
	vkQueueWaitIdle( graphicsQueue );
	vkFreeCommandBuffers( device, oneTimePool, 1, &cmd );
}

VkShaderModule idVulkanRayPipelineManager::LoadShaderModule( const uint32_t *spv, uint32_t bytes ) {
	VkShaderModuleCreateInfo ci = {};
	ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	ci.codeSize = bytes;
	ci.pCode    = spv;
	VkShaderModule mod;
	if ( vkCreateShaderModule( device, &ci, NULL, &mod ) != VK_SUCCESS ) {
		return VK_NULL_HANDLE;
	}
	return mod;
}

// ---------------------------------------------------------------------------

bool idVulkanRayPipelineManager::CreateOutputImage( uint32_t width, uint32_t height ) {
	VkImageCreateInfo ici = {};
	ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	ici.imageType     = VK_IMAGE_TYPE_2D;
	ici.format        = VK_FORMAT_R8G8B8A8_UNORM;
	ici.extent        = { width, height, 1 };
	ici.mipLevels     = 1;
	ici.arrayLayers   = 1;
	ici.samples       = VK_SAMPLE_COUNT_1_BIT;
	ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
	ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

	if ( vkCreateImage( device, &ici, NULL, &outputImage ) != VK_SUCCESS ) {
		common->Warning( "RayPipeline: failed to create output image" );
		return false;
	}

	VkMemoryRequirements mr;
	vkGetImageMemoryRequirements( device, outputImage, &mr );

	VkMemoryAllocateInfo mai = {};
	mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	mai.allocationSize  = mr.size;
	mai.memoryTypeIndex = FindMemoryType( mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT );
	if ( vkAllocateMemory( device, &mai, NULL, &outputMemory ) != VK_SUCCESS ) {
		common->Warning( "RayPipeline: failed to allocate output image memory" );
		return false;
	}
	vkBindImageMemory( device, outputImage, outputMemory, 0 );

	VkImageViewCreateInfo vci = {};
	vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	vci.image    = outputImage;
	vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
	vci.format   = VK_FORMAT_R8G8B8A8_UNORM;
	vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
	if ( vkCreateImageView( device, &vci, NULL, &outputView ) != VK_SUCCESS ) {
		common->Warning( "RayPipeline: failed to create output image view" );
		return false;
	}

	// Transition to GENERAL layout for storage image use
	SubmitOneTime( [&]( VkCommandBuffer cmd ) {
		VkImageMemoryBarrier barrier = {};
		barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
		barrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image               = outputImage;
		barrier.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		barrier.srcAccessMask       = 0;
		barrier.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
		vkCmdPipelineBarrier( cmd,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
			0, 0, NULL, 0, NULL, 1, &barrier );
	} );

	return true;
}

bool idVulkanRayPipelineManager::CreateDescriptorSetLayout() {
	// Binding 0: acceleration structure (TLAS)
	// Binding 1: storage image (output)
	// Binding 2: camera UBO
	// Binding 3: lights UBO
	// Binding 4: benchmark SSBO (always present in layout; dummy buffer if disabled)
	VkDescriptorSetLayoutBinding bindings[5] = {};

	bindings[0].binding         = 0;
	bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	bindings[0].descriptorCount = 1;
	bindings[0].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

	bindings[1].binding         = 1;
	bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	bindings[1].descriptorCount = 1;
	bindings[1].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

	bindings[2].binding         = 2;
	bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	bindings[2].descriptorCount = 1;
	bindings[2].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

	bindings[3].binding         = 3;
	bindings[3].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	bindings[3].descriptorCount = 1;
	bindings[3].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

	bindings[4].binding         = 4;
	bindings[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	bindings[4].descriptorCount = 1;
	bindings[4].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

	VkDescriptorSetLayoutCreateInfo ci = {};
	ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	ci.bindingCount = 5;
	ci.pBindings    = bindings;

	if ( vkCreateDescriptorSetLayout( device, &ci, NULL, &descSetLayout ) != VK_SUCCESS ) {
		common->Warning( "RayPipeline: failed to create descriptor set layout" );
		return false;
	}
	return true;
}

bool idVulkanRayPipelineManager::CreateDescriptorPool() {
	VkDescriptorPoolSize poolSizes[4] = {};
	poolSizes[0].type            = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	poolSizes[0].descriptorCount = 1;
	poolSizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	poolSizes[1].descriptorCount = 1;
	poolSizes[2].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	poolSizes[2].descriptorCount = 2;
	poolSizes[3].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	poolSizes[3].descriptorCount = 1;

	VkDescriptorPoolCreateInfo ci = {};
	ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	ci.maxSets       = 1;
	ci.poolSizeCount = 4;
	ci.pPoolSizes    = poolSizes;

	if ( vkCreateDescriptorPool( device, &ci, NULL, &descPool ) != VK_SUCCESS ) {
		common->Warning( "RayPipeline: failed to create descriptor pool" );
		return false;
	}
	return true;
}

bool idVulkanRayPipelineManager::CreateDescriptorSet() {
	VkDescriptorSetAllocateInfo ai = {};
	ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	ai.descriptorPool     = descPool;
	ai.descriptorSetCount = 1;
	ai.pSetLayouts        = &descSetLayout;

	if ( vkAllocateDescriptorSets( device, &ai, &descSet ) != VK_SUCCESS ) {
		common->Warning( "RayPipeline: failed to allocate descriptor set" );
		return false;
	}
	return true;
}

bool idVulkanRayPipelineManager::CreatePipeline() {
	// Shader modules
	VkShaderModule raygen = LoadShaderModule( rt_raygen_rgen_spv,       sizeof(rt_raygen_rgen_spv) );
	VkShaderModule miss   = LoadShaderModule( rt_miss_rmiss_spv,         sizeof(rt_miss_rmiss_spv) );
	VkShaderModule smiss  = LoadShaderModule( rt_shadow_miss_rmiss_spv,  sizeof(rt_shadow_miss_rmiss_spv) );
	VkShaderModule hit    = LoadShaderModule( rt_hit_rchit_spv,          sizeof(rt_hit_rchit_spv) );

	if ( raygen == VK_NULL_HANDLE || miss == VK_NULL_HANDLE ||
	     smiss  == VK_NULL_HANDLE || hit  == VK_NULL_HANDLE ) {
		common->Warning( "RayPipeline: failed to load shader modules" );
		if ( raygen ) vkDestroyShaderModule( device, raygen, NULL );
		if ( miss )   vkDestroyShaderModule( device, miss,   NULL );
		if ( smiss )  vkDestroyShaderModule( device, smiss,  NULL );
		if ( hit )    vkDestroyShaderModule( device, hit,    NULL );
		return false;
	}

	// Shader stages
	// Index 0: raygen, 1: primary miss, 2: shadow miss, 3: closest hit
	VkPipelineShaderStageCreateInfo stages[4] = {};

	stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage  = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
	stages[0].module = raygen;
	stages[0].pName  = "main";

	stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage  = VK_SHADER_STAGE_MISS_BIT_KHR;
	stages[1].module = miss;
	stages[1].pName  = "main";

	stages[2].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[2].stage  = VK_SHADER_STAGE_MISS_BIT_KHR;
	stages[2].module = smiss;
	stages[2].pName  = "main";

	stages[3].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[3].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	stages[3].module = hit;
	stages[3].pName  = "main";

	// Shader groups
	// Group 0: raygen
	// Group 1: primary miss
	// Group 2: shadow miss
	// Group 3: closest hit
	VkRayTracingShaderGroupCreateInfoKHR groups[4] = {};

	groups[0].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	groups[0].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	groups[0].generalShader      = 0;
	groups[0].closestHitShader   = VK_SHADER_UNUSED_KHR;
	groups[0].anyHitShader       = VK_SHADER_UNUSED_KHR;
	groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;

	groups[1].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	groups[1].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	groups[1].generalShader      = 1;
	groups[1].closestHitShader   = VK_SHADER_UNUSED_KHR;
	groups[1].anyHitShader       = VK_SHADER_UNUSED_KHR;
	groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;

	groups[2].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	groups[2].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	groups[2].generalShader      = 2;
	groups[2].closestHitShader   = VK_SHADER_UNUSED_KHR;
	groups[2].anyHitShader       = VK_SHADER_UNUSED_KHR;
	groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

	groups[3].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	groups[3].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
	groups[3].generalShader      = VK_SHADER_UNUSED_KHR;
	groups[3].closestHitShader   = 3;
	groups[3].anyHitShader       = VK_SHADER_UNUSED_KHR;
	groups[3].intersectionShader = VK_SHADER_UNUSED_KHR;

	// Pipeline layout
	VkPipelineLayoutCreateInfo layoutCI = {};
	layoutCI.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layoutCI.setLayoutCount = 1;
	layoutCI.pSetLayouts    = &descSetLayout;

	if ( vkCreatePipelineLayout( device, &layoutCI, NULL, &pipelineLayout ) != VK_SUCCESS ) {
		common->Warning( "RayPipeline: failed to create pipeline layout" );
		for ( auto &s : stages ) vkDestroyShaderModule( device, s.module, NULL );
		return false;
	}

	// RT pipeline
	VkRayTracingPipelineCreateInfoKHR rtCI = {};
	rtCI.sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
	rtCI.stageCount                   = 4;
	rtCI.pStages                      = stages;
	rtCI.groupCount                   = 4;
	rtCI.pGroups                      = groups;
	rtCI.maxPipelineRayRecursionDepth = 2;
	rtCI.layout                       = pipelineLayout;

	VkResult r = pfn_vkCreateRayTracingPipelinesKHR( device, VK_NULL_HANDLE,
		VK_NULL_HANDLE, 1, &rtCI, NULL, &rtPipeline );

	for ( auto &s : stages ) vkDestroyShaderModule( device, s.module, NULL );

	if ( r != VK_SUCCESS ) {
		common->Warning( "RayPipeline: vkCreateRayTracingPipelinesKHR failed: %d", (int)r );
		return false;
	}
	return true;
}

#if ANDROID
extern PFN_vkGetPhysicalDeviceProperties2 p_vkGetPhysicalDeviceProperties2;
#endif

bool idVulkanRayPipelineManager::CreateSBT() {
#if ANDROID
    if (p_vkGetPhysicalDeviceProperties2 == nullptr){
        return false;
    }
#endif
	// Query RT pipeline properties for handle size and alignment
	VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps = {};
	rtProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;

	VkPhysicalDeviceProperties2 props2 = {};
	props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	props2.pNext = &rtProps;
#ifndef ANDROID
	vkGetPhysicalDeviceProperties2( physicalDevice, &props2 );
#else
    p_vkGetPhysicalDeviceProperties2(physicalDevice, &props2);
#endif

	uint32_t handleSize      = rtProps.shaderGroupHandleSize;
	uint32_t handleAlignment = rtProps.shaderGroupHandleAlignment;
	uint32_t baseAlignment   = rtProps.shaderGroupBaseAlignment;

	// Align each entry to handleAlignment, and each region to baseAlignment
	auto alignUp = []( uint32_t v, uint32_t a ) { return (v + a - 1) & ~(a - 1); };

	uint32_t handleAligned = alignUp( handleSize, handleAlignment );
	uint32_t regionAligned = alignUp( handleAligned, baseAlignment );

	// 4 groups: raygen (0), primary miss (1), shadow miss (2), closest hit (3)
	uint32_t groupCount = 4;
	std::vector<uint8_t> handles( handleSize * groupCount );
	if ( pfn_vkGetRayTracingShaderGroupHandlesKHR( device, rtPipeline,
		0, groupCount, handles.size(), handles.data() ) != VK_SUCCESS ) {
		common->Warning( "RayPipeline: failed to get SBT handles" );
		return false;
	}

	// SBT layout:
	//   [raygen region]     1 entry
	//   [miss region]       2 entries (primary + shadow)
	//   [hit region]        1 entry
	uint32_t raygenStride = regionAligned;
	uint32_t missStride   = handleAligned;
	uint32_t hitStride    = handleAligned;

	VkDeviceSize raygenSize  = regionAligned;
	VkDeviceSize missSize    = alignUp( 2 * handleAligned, baseAlignment );
	VkDeviceSize hitSize     = regionAligned;
	VkDeviceSize totalSize   = raygenSize + missSize + hitSize;

	// Allocate host-visible SBT buffer
	VkBufferCreateInfo bci = {};
	bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bci.size        = totalSize;
	bci.usage       = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
	                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	if ( vkCreateBuffer( device, &bci, NULL, &sbtBuffer ) != VK_SUCCESS ) {
		common->Warning( "RayPipeline: failed to create SBT buffer" );
		return false;
	}

	VkMemoryRequirements mr;
	vkGetBufferMemoryRequirements( device, sbtBuffer, &mr );

	VkMemoryAllocateFlagsInfo flagsInfo = {};
	flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
	flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

	VkPhysicalDeviceMemoryProperties memProp;
	vkGetPhysicalDeviceMemoryProperties( physicalDevice, &memProp );
	uint32_t memIdx = UINT32_MAX;
	for ( uint32_t i = 0; i < memProp.memoryTypeCount; i++ ) {
		if ( (mr.memoryTypeBits & (1u<<i)) &&
		     (memProp.memoryTypes[i].propertyFlags &
		      (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
		      (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ) {
			memIdx = i; break;
		}
	}

	VkMemoryAllocateInfo mai = {};
	mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	mai.pNext           = &flagsInfo;
	mai.allocationSize  = mr.size;
	mai.memoryTypeIndex = memIdx;

	if ( vkAllocateMemory( device, &mai, NULL, &sbtMemory ) != VK_SUCCESS ) {
		common->Warning( "RayPipeline: failed to allocate SBT memory" );
		return false;
	}
	vkBindBufferMemory( device, sbtBuffer, sbtMemory, 0 );

	// Write handles into the SBT
	uint8_t *sbtData;
	vkMapMemory( device, sbtMemory, 0, totalSize, 0, (void**)&sbtData );
	memset( sbtData, 0, totalSize );

	// Raygen: group 0
	memcpy( sbtData, handles.data() + 0 * handleSize, handleSize );
	// Miss 0 (primary): group 1
	memcpy( sbtData + raygenSize, handles.data() + 1 * handleSize, handleSize );
	// Miss 1 (shadow):  group 2
	memcpy( sbtData + raygenSize + handleAligned, handles.data() + 2 * handleSize, handleSize );
	// Hit:              group 3
	memcpy( sbtData + raygenSize + missSize, handles.data() + 3 * handleSize, handleSize );

	vkUnmapMemory( device, sbtMemory );

	// Compute device addresses for the SBT regions
	VkBufferDeviceAddressInfo addrInfo = {};
	addrInfo.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	addrInfo.buffer = sbtBuffer;
	VkDeviceAddress base = pfn_vkGetBufferDeviceAddressKHR( device, &addrInfo );

	raygenRegion.deviceAddress = base;
	raygenRegion.stride        = raygenStride;
	raygenRegion.size          = raygenSize;

	missRegion.deviceAddress = base + raygenSize;
	missRegion.stride        = missStride;
	missRegion.size          = missSize;

	hitRegion.deviceAddress = base + raygenSize + missSize;
	hitRegion.stride        = hitStride;
	hitRegion.size          = hitSize;

	memset( &callableRegion, 0, sizeof(callableRegion) );
	return true;
}

// ---------------------------------------------------------------------------
// Per-frame
// ---------------------------------------------------------------------------

void idVulkanRayPipelineManager::UpdateDescriptors(
	VkAccelerationStructureKHR tlas,
	VkBuffer cameraUBO, VkDeviceSize cameraOffset,
	VkBuffer lightsUBO, VkDeviceSize lightsOffset )
{
	VkWriteDescriptorSetAccelerationStructureKHR asWrite = {};
	asWrite.sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
	asWrite.accelerationStructureCount = 1;
	asWrite.pAccelerationStructures    = &tlas;

	VkDescriptorBufferInfo camInfo = {};
	camInfo.buffer = cameraUBO;
	camInfo.offset = cameraOffset;
	camInfo.range  = sizeof(CameraUBO);

	VkDescriptorBufferInfo lightInfo = {};
	lightInfo.buffer = lightsUBO;
	lightInfo.offset = lightsOffset;
	lightInfo.range  = sizeof(LightsUBO);

	VkDescriptorBufferInfo benchInfo = {};
	benchInfo.buffer = benchmarkBuffer;
	benchInfo.offset = 0;
	benchInfo.range  = VK_WHOLE_SIZE;

	VkWriteDescriptorSet writes[4] = {};

	writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[0].pNext           = &asWrite;
	writes[0].dstSet          = descSet;
	writes[0].dstBinding      = 0;
	writes[0].descriptorCount = 1;
	writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

	writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[1].dstSet          = descSet;
	writes[1].dstBinding      = 2;
	writes[1].descriptorCount = 1;
	writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writes[1].pBufferInfo     = &camInfo;

	writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[2].dstSet          = descSet;
	writes[2].dstBinding      = 3;
	writes[2].descriptorCount = 1;
	writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writes[2].pBufferInfo     = &lightInfo;

	writes[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[3].dstSet          = descSet;
	writes[3].dstBinding      = 4;
	writes[3].descriptorCount = 1;
	writes[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writes[3].pBufferInfo     = &benchInfo;

	vkUpdateDescriptorSets( device, 4, writes, 0, NULL );
}

void idVulkanRayPipelineManager::BindAndTrace( VkCommandBuffer cmd,
                                               uint32_t width, uint32_t height ) {
	vkCmdBindPipeline( cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline );
	vkCmdBindDescriptorSets( cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
		pipelineLayout, 0, 1, &descSet, 0, NULL );
	pfn_vkCmdTraceRaysKHR( cmd,
		&raygenRegion, &missRegion, &hitRegion, &callableRegion,
		width, height, 1 );
}

void idVulkanRayPipelineManager::BlitToSwapchain( VkCommandBuffer cmd,
                                                   VkImage swapchainImage,
                                                   VkImageLayout swapchainInitialLayout,
                                                   uint32_t width, uint32_t height ) {
	// Barrier: RT shader write → transfer read
	VkImageMemoryBarrier outToSrc = {};
	outToSrc.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	outToSrc.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
	outToSrc.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	outToSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	outToSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	outToSrc.image               = outputImage;
	outToSrc.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
	outToSrc.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
	outToSrc.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;

	VkImageMemoryBarrier swapToDst = {};
	swapToDst.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	swapToDst.oldLayout           = swapchainInitialLayout;
	swapToDst.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	swapToDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	swapToDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	swapToDst.image               = swapchainImage;
	swapToDst.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
	swapToDst.srcAccessMask       = 0;
	swapToDst.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;

	VkImageMemoryBarrier preCopy[2] = { outToSrc, swapToDst };
	vkCmdPipelineBarrier( cmd,
		VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		0, 0, NULL, 0, NULL, 2, preCopy );

	// Copy output image to swapchain (1:1 — same resolution)
	VkImageCopy copyRegion = {};
	copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
	copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
	copyRegion.extent         = { width, height, 1 };
	vkCmdCopyImage( cmd,
		outputImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		1, &copyRegion );

	// Barriers: swapchain → PRESENT_SRC, output → GENERAL
	VkImageMemoryBarrier swapToPresent = {};
	swapToPresent.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	swapToPresent.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	swapToPresent.newLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	swapToPresent.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	swapToPresent.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	swapToPresent.image               = swapchainImage;
	swapToPresent.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
	swapToPresent.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
	swapToPresent.dstAccessMask       = 0;

	VkImageMemoryBarrier outBackToGeneral = {};
	outBackToGeneral.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	outBackToGeneral.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	outBackToGeneral.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
	outBackToGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	outBackToGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	outBackToGeneral.image               = outputImage;
	outBackToGeneral.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
	outBackToGeneral.srcAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
	outBackToGeneral.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;

	VkImageMemoryBarrier postCopy[2] = { swapToPresent, outBackToGeneral };
	vkCmdPipelineBarrier( cmd,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT |
		VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
		0, 0, NULL, 0, NULL, 2, postCopy );
}

// ---------------------------------------------------------------------------
// Benchmark SSBO
// ---------------------------------------------------------------------------

bool idVulkanRayPipelineManager::CreateBenchmarkBuffer() {
	// In benchmark mode, allocate MAX_BENCH_RECORDS records + 16-byte header.
	// Otherwise allocate a minimal 256-byte stub so binding 4 is always valid.
	VkDeviceSize bufSize = com_benchmark.GetBool()
		? (VkDeviceSize)16 + MAX_BENCH_RECORDS * sizeof(TrainingRecord)
		: 256;

	VkBufferCreateInfo bci = {};
	bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bci.size        = bufSize;
	bci.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	if ( vkCreateBuffer( device, &bci, NULL, &benchmarkBuffer ) != VK_SUCCESS ) {
		return false;
	}

	VkMemoryRequirements mr;
	vkGetBufferMemoryRequirements( device, benchmarkBuffer, &mr );

	VkMemoryAllocateInfo mai = {};
	mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	mai.allocationSize  = mr.size;
	mai.memoryTypeIndex = FindMemoryType( mr.memoryTypeBits,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT );
	if ( mai.memoryTypeIndex == UINT32_MAX ) {
		vkDestroyBuffer( device, benchmarkBuffer, NULL );
		benchmarkBuffer = VK_NULL_HANDLE;
		return false;
	}

	if ( vkAllocateMemory( device, &mai, NULL, &benchmarkMemory ) != VK_SUCCESS ) {
		vkDestroyBuffer( device, benchmarkBuffer, NULL );
		benchmarkBuffer = VK_NULL_HANDLE;
		return false;
	}
	vkBindBufferMemory( device, benchmarkBuffer, benchmarkMemory, 0 );

	// Persistent map (HOST_COHERENT: no explicit flush needed)
	if ( vkMapMemory( device, benchmarkMemory, 0, bufSize, 0, &benchmarkMapped ) != VK_SUCCESS ) {
		vkFreeMemory( device, benchmarkMemory, NULL );  benchmarkMemory = VK_NULL_HANDLE;
		vkDestroyBuffer( device, benchmarkBuffer, NULL ); benchmarkBuffer = VK_NULL_HANDLE;
		return false;
	}

	// Zero out (sets recordCount = 0 and clears padding)
	memset( benchmarkMapped, 0, (size_t)bufSize );
	return true;
}

void idVulkanRayPipelineManager::ResetBenchmarkCounter() {
	if ( benchmarkMapped ) {
		// First uint32 in the buffer is the record count
		((uint32_t*)benchmarkMapped)[0] = 0u;
		// HOST_COHERENT: no flush needed
	}
}

void idVulkanRayPipelineManager::ReadBenchmarkRecords(
	uint32_t &outCount, const TrainingRecord *&outRecords )
{
	if ( !benchmarkMapped ) {
		outCount   = 0;
		outRecords = nullptr;
		return;
	}
	outCount = ((uint32_t*)benchmarkMapped)[0];
	if ( outCount > MAX_BENCH_RECORDS ) outCount = MAX_BENCH_RECORDS;
	outRecords = (const TrainingRecord*)((char*)benchmarkMapped + 16);
}

#endif // ID_VULKAN
