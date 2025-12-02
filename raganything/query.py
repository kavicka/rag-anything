"""
Query functionality for RAGAnything

Contains all query-related methods for both text and multimodal queries
"""

import json
import hashlib
import re
import math
from typing import Dict, List, Any
from pathlib import Path
from lightrag import QueryParam
from lightrag.utils import always_get_an_event_loop
from raganything.prompt import PROMPTS
from raganything.utils import (
    get_processor_for_type,
    encode_image_to_base64,
    validate_image_file,
    resolve_image_path,
)

# Try to import numpy for vector operations, fallback to manual calculation
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class QueryMixin:
    """QueryMixin class containing query functionality for RAGAnything"""

    def _generate_multimodal_cache_key(
        self, query: str, multimodal_content: List[Dict[str, Any]], mode: str, **kwargs
    ) -> str:
        """
        Generate cache key for multimodal query

        Args:
            query: Base query text
            multimodal_content: List of multimodal content
            mode: Query mode
            **kwargs: Additional parameters

        Returns:
            str: Cache key hash
        """
        # Create a normalized representation of the query parameters
        cache_data = {
            "query": query.strip(),
            "mode": mode,
        }

        # Normalize multimodal content for stable caching
        normalized_content = []
        if multimodal_content:
            for item in multimodal_content:
                if isinstance(item, dict):
                    normalized_item = {}
                    for key, value in item.items():
                        # For file paths, use basename to make cache more portable
                        if key in [
                            "img_path",
                            "image_path",
                            "file_path",
                        ] and isinstance(value, str):
                            normalized_item[key] = Path(value).name
                        # For large content, create a hash instead of storing directly
                        elif (
                            key in ["table_data", "table_body"]
                            and isinstance(value, str)
                            and len(value) > 200
                        ):
                            normalized_item[f"{key}_hash"] = hashlib.md5(
                                value.encode()
                            ).hexdigest()
                        else:
                            normalized_item[key] = value
                    normalized_content.append(normalized_item)
                else:
                    normalized_content.append(item)

        cache_data["multimodal_content"] = normalized_content

        # Add relevant kwargs to cache data
        relevant_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "stream",
                "response_type",
                "top_k",
                "max_tokens",
                "temperature",
                # "only_need_context",
                # "only_need_prompt",
            ]
        }
        cache_data.update(relevant_kwargs)

        # Generate hash from the cache data
        cache_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest()

        return f"multimodal_query:{cache_hash}"

    async def aquery(
        self, query: str, mode: str = "mix", system_prompt: str | None = None, **kwargs
    ) -> str:
        """
        Pure text query - directly calls LightRAG's query functionality

        Args:
            query: Query text
            mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
            system_prompt: Optional system prompt to include.
            **kwargs: Other query parameters, will be passed to QueryParam
                - vlm_enhanced: bool, default True when vision_model_func is available.
                  If True, will parse image paths in retrieved context and replace them
                  with base64 encoded images for VLM processing.

        Returns:
            str: Query result
        """
        if self.lightrag is None:
            raise ValueError(
                "No LightRAG instance available. Please process documents first or provide a pre-initialized LightRAG instance."
            )

        # Check if VLM enhanced query should be used
        vlm_enhanced = kwargs.pop("vlm_enhanced", None)

        # Auto-determine VLM enhanced based on availability
        if vlm_enhanced is None:
            vlm_enhanced = (
                hasattr(self, "vision_model_func")
                and self.vision_model_func is not None
            )

        # Use VLM enhanced query if enabled and available
        if (
            vlm_enhanced
            and hasattr(self, "vision_model_func")
            and self.vision_model_func
        ):
            return await self.aquery_vlm_enhanced(
                query, mode=mode, system_prompt=system_prompt, **kwargs
            )
        elif vlm_enhanced and (
            not hasattr(self, "vision_model_func") or not self.vision_model_func
        ):
            self.logger.warning(
                "VLM enhanced query requested but vision_model_func is not available, falling back to normal query"
            )

        # Create query parameters
        query_param = QueryParam(mode=mode, **kwargs)

        self.logger.info(f"Executing text query: {query[:100]}...")
        self.logger.info(f"Query mode: {mode}")

        # Call LightRAG's query method
        result = await self.lightrag.aquery(
            query, param=query_param, system_prompt=system_prompt
        )

        self.logger.info("Text query completed")
        return result

    async def aquery_with_multimodal(
        self,
        query: str,
        multimodal_content: List[Dict[str, Any]] = None,
        mode: str = "mix",
        **kwargs,
    ) -> str:
        """
        Multimodal query - combines text and multimodal content for querying

        Args:
            query: Base query text
            multimodal_content: List of multimodal content, each element contains:
                - type: Content type ("image", "table", "equation", etc.)
                - Other fields depend on type (e.g., img_path, table_data, latex, etc.)
            mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
            **kwargs: Other query parameters, will be passed to QueryParam

        Returns:
            str: Query result

        Examples:
            # Pure text query
            result = await rag.query_with_multimodal("What is machine learning?")

            # Image query
            result = await rag.query_with_multimodal(
                "Analyze the content in this image",
                multimodal_content=[{
                    "type": "image",
                    "img_path": "./image.jpg"
                }]
            )

            # Table query
            result = await rag.query_with_multimodal(
                "Analyze the data trends in this table",
                multimodal_content=[{
                    "type": "table",
                    "table_data": "Name,Age\nAlice,25\nBob,30"
                }]
            )
        """
        # Ensure LightRAG is initialized
        await self._ensure_lightrag_initialized()

        self.logger.info(f"Executing multimodal query: {query[:100]}...")
        self.logger.info(f"Query mode: {mode}")

        # If no multimodal content, fallback to pure text query
        if not multimodal_content:
            self.logger.info("No multimodal content provided, executing text query")
            return await self.aquery(query, mode=mode, **kwargs)

        # Generate cache key for multimodal query
        cache_key = self._generate_multimodal_cache_key(
            query, multimodal_content, mode, **kwargs
        )

        # Check cache if available and enabled
        cached_result = None
        if (
            hasattr(self, "lightrag")
            and self.lightrag
            and hasattr(self.lightrag, "llm_response_cache")
            and self.lightrag.llm_response_cache
        ):
            if self.lightrag.llm_response_cache.global_config.get(
                "enable_llm_cache", True
            ):
                try:
                    cached_result = await self.lightrag.llm_response_cache.get_by_id(
                        cache_key
                    )
                    if cached_result and isinstance(cached_result, dict):
                        result_content = cached_result.get("return")
                        if result_content:
                            self.logger.info(
                                f"Multimodal query cache hit: {cache_key[:16]}..."
                            )
                            return result_content
                except Exception as e:
                    self.logger.debug(f"Error accessing multimodal query cache: {e}")

        # Process multimodal content to generate enhanced query text
        enhanced_query = await self._process_multimodal_query_content(
            query, multimodal_content
        )

        self.logger.info(
            f"Generated enhanced query length: {len(enhanced_query)} characters"
        )

        # Execute enhanced query
        result = await self.aquery(enhanced_query, mode=mode, **kwargs)

        # Save to cache if available and enabled
        if (
            hasattr(self, "lightrag")
            and self.lightrag
            and hasattr(self.lightrag, "llm_response_cache")
            and self.lightrag.llm_response_cache
        ):
            if self.lightrag.llm_response_cache.global_config.get(
                "enable_llm_cache", True
            ):
                try:
                    # Create cache entry for multimodal query
                    cache_entry = {
                        "return": result,
                        "cache_type": "multimodal_query",
                        "original_query": query,
                        "multimodal_content_count": len(multimodal_content),
                        "mode": mode,
                    }

                    await self.lightrag.llm_response_cache.upsert(
                        {cache_key: cache_entry}
                    )
                    self.logger.info(
                        f"Saved multimodal query result to cache: {cache_key[:16]}..."
                    )
                except Exception as e:
                    self.logger.debug(f"Error saving multimodal query to cache: {e}")

        # Ensure cache is persisted to disk
        if (
            hasattr(self, "lightrag")
            and self.lightrag
            and hasattr(self.lightrag, "llm_response_cache")
            and self.lightrag.llm_response_cache
        ):
            try:
                await self.lightrag.llm_response_cache.index_done_callback()
            except Exception as e:
                self.logger.debug(f"Error persisting multimodal query cache: {e}")

        self.logger.info("Multimodal query completed")
        return result

    async def aquery_vlm_enhanced(
        self, query: str, mode: str = "mix", system_prompt: str | None = None, **kwargs
    ) -> str:
        """
        VLM enhanced query - replaces image paths in retrieved context with base64 encoded images for VLM processing

        Args:
            query: User query
            mode: Underlying LightRAG query mode
            system_prompt: Optional system prompt to include
            **kwargs: Other query parameters

        Returns:
            str: VLM query result
        """
        # Ensure VLM is available
        if not hasattr(self, "vision_model_func") or not self.vision_model_func:
            raise ValueError(
                "VLM enhanced query requires vision_model_func. "
                "Please provide a vision model function when initializing RAGAnything."
            )

        # Ensure LightRAG is initialized
        await self._ensure_lightrag_initialized()

        self.logger.info(f"Executing VLM enhanced query: {query[:100]}...")

        # Clear previous image cache
        if hasattr(self, "_current_images_base64"):
            delattr(self, "_current_images_base64")

        # 1. Get original retrieval prompt (without generating final answer)
        query_param = QueryParam(mode=mode, only_need_prompt=True, **kwargs)
        raw_prompt = await self.lightrag.aquery(query, param=query_param)

        self.logger.debug("Retrieved raw prompt from LightRAG")

        # 2. Extract and process image paths
        enhanced_prompt, images_found = await self._process_image_paths_for_vlm(
            raw_prompt
        )

        if not images_found:
            self.logger.info("No valid images found, falling back to normal query")
            # Fallback to normal query
            query_param = QueryParam(mode=mode, **kwargs)
            return await self.lightrag.aquery(
                query, param=query_param, system_prompt=system_prompt
            )

        self.logger.info(f"Processed {images_found} images for VLM")

        # 3. Build VLM message format
        messages = self._build_vlm_messages_with_images(
            enhanced_prompt, query, system_prompt
        )

        # 4. Call VLM for question answering
        result = await self._call_vlm_with_multimodal_content(messages)

        self.logger.info("VLM enhanced query completed")
        return result

    async def _process_multimodal_query_content(
        self, base_query: str, multimodal_content: List[Dict[str, Any]]
    ) -> str:
        """
        Process multimodal query content to generate enhanced query text

        Args:
            base_query: Base query text
            multimodal_content: List of multimodal content

        Returns:
            str: Enhanced query text
        """
        self.logger.info("Starting multimodal query content processing...")

        enhanced_parts = [f"User query: {base_query}"]

        for i, content in enumerate(multimodal_content):
            content_type = content.get("type", "unknown")
            self.logger.info(
                f"Processing {i+1}/{len(multimodal_content)} multimodal content: {content_type}"
            )

            try:
                # Get appropriate processor
                processor = get_processor_for_type(self.modal_processors, content_type)

                if processor:
                    # Generate content description
                    description = await self._generate_query_content_description(
                        processor, content, content_type
                    )
                    enhanced_parts.append(
                        f"\nRelated {content_type} content: {description}"
                    )
                else:
                    # If no appropriate processor, use basic description
                    basic_desc = str(content)[:200]
                    enhanced_parts.append(
                        f"\nRelated {content_type} content: {basic_desc}"
                    )

            except Exception as e:
                self.logger.error(f"Error processing multimodal content: {str(e)}")
                # Continue processing other content
                continue

        enhanced_query = "\n".join(enhanced_parts)
        enhanced_query += PROMPTS["QUERY_ENHANCEMENT_SUFFIX"]

        self.logger.info("Multimodal query content processing completed")
        return enhanced_query

    async def _generate_query_content_description(
        self, processor, content: Dict[str, Any], content_type: str
    ) -> str:
        """
        Generate content description for query

        Args:
            processor: Multimodal processor
            content: Content data
            content_type: Content type

        Returns:
            str: Content description
        """
        try:
            if content_type == "image":
                return await self._describe_image_for_query(processor, content)
            elif content_type == "table":
                return await self._describe_table_for_query(processor, content)
            elif content_type == "equation":
                return await self._describe_equation_for_query(processor, content)
            else:
                return await self._describe_generic_for_query(
                    processor, content, content_type
                )

        except Exception as e:
            self.logger.error(f"Error generating {content_type} description: {str(e)}")
            return f"{content_type} content: {str(content)[:100]}"

    async def _describe_image_for_query(
        self, processor, content: Dict[str, Any]
    ) -> str:
        """Generate image description for query"""
        image_path = content.get("img_path")
        captions = content.get("image_caption", content.get("img_caption", []))
        footnotes = content.get("image_footnote", content.get("img_footnote", []))

        if image_path and Path(image_path).exists():
            # If image exists, use vision model to generate description
            image_base64 = processor._encode_image_to_base64(image_path)
            if image_base64:
                prompt = PROMPTS["QUERY_IMAGE_DESCRIPTION"]
                description = await processor.modal_caption_func(
                    prompt,
                    image_data=image_base64,
                    system_prompt=PROMPTS["QUERY_IMAGE_ANALYST_SYSTEM"],
                )
                return description

        # If image doesn't exist or processing failed, use existing information
        parts = []
        if image_path:
            parts.append(f"Image path: {image_path}")
        if captions:
            parts.append(f"Image captions: {', '.join(captions)}")
        if footnotes:
            parts.append(f"Image footnotes: {', '.join(footnotes)}")

        return "; ".join(parts) if parts else "Image content information incomplete"

    async def _describe_table_for_query(
        self, processor, content: Dict[str, Any]
    ) -> str:
        """Generate table description for query"""
        table_data = content.get("table_data", "")
        table_caption = content.get("table_caption", "")

        prompt = PROMPTS["QUERY_TABLE_ANALYSIS"].format(
            table_data=table_data, table_caption=table_caption
        )

        description = await processor.modal_caption_func(
            prompt, system_prompt=PROMPTS["QUERY_TABLE_ANALYST_SYSTEM"]
        )

        return description

    async def _describe_equation_for_query(
        self, processor, content: Dict[str, Any]
    ) -> str:
        """Generate equation description for query"""
        latex = content.get("latex", "")
        equation_caption = content.get("equation_caption", "")

        prompt = PROMPTS["QUERY_EQUATION_ANALYSIS"].format(
            latex=latex, equation_caption=equation_caption
        )

        description = await processor.modal_caption_func(
            prompt, system_prompt=PROMPTS["QUERY_EQUATION_ANALYST_SYSTEM"]
        )

        return description

    async def _describe_generic_for_query(
        self, processor, content: Dict[str, Any], content_type: str
    ) -> str:
        """Generate generic content description for query"""
        content_str = str(content)

        prompt = PROMPTS["QUERY_GENERIC_ANALYSIS"].format(
            content_type=content_type, content_str=content_str
        )

        description = await processor.modal_caption_func(
            prompt,
            system_prompt=PROMPTS["QUERY_GENERIC_ANALYST_SYSTEM"].format(
                content_type=content_type
            ),
        )

        return description

    async def _process_image_paths_for_vlm(self, prompt: str) -> tuple[str, int]:
        """
        Process image paths in prompt, keeping original paths and adding VLM markers

        Args:
            prompt: Original prompt

        Returns:
            tuple: (processed prompt, image count)
        """
        enhanced_prompt = prompt
        images_processed = 0

        # Initialize image cache
        self._current_images_base64 = []

        # Enhanced regex pattern for matching image paths
        # Matches only the path ending with image file extensions
        image_path_pattern = (
            r"Image Path:\s*([^\r\n]*?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif))"
        )

        # First, let's see what matches we find
        matches = re.findall(image_path_pattern, prompt)
        self.logger.info(f"Found {len(matches)} image path matches in prompt")

        def replace_image_path(match):
            nonlocal images_processed

            image_path = match.group(1).strip()
            self.logger.debug(f"Processing image path: '{image_path}'")

            # Validate path format (basic check)
            if not image_path or len(image_path) < 3:
                self.logger.warning(f"Invalid image path format: {image_path}")
                return match.group(0)  # Keep original

            # Try to resolve the image path using output directory if available
            output_dir = getattr(self.config, 'parser_output_dir', None)
            
            # Use utility function to validate image file with output_dir for path resolution
            # validate_image_file will internally call resolve_image_path
            self.logger.debug(f"Calling validate_image_file for: {image_path} with output_dir: {output_dir}")
            is_valid = validate_image_file(image_path, output_dir=output_dir)
            
            # Get the resolved path for encoding
            resolved_path = resolve_image_path(image_path, output_dir)
            self.logger.debug(f"Validation result for {image_path}: {is_valid}, resolved path: {resolved_path}")

            if not is_valid:
                self.logger.warning(f"Image validation failed for: {image_path} (resolved: {resolved_path})")
                return match.group(0)  # Keep original if validation fails

            try:
                # Encode image to base64 using utility function (use resolved path)
                self.logger.debug(f"Attempting to encode image: {resolved_path}")
                image_base64 = encode_image_to_base64(resolved_path)
                if image_base64:
                    images_processed += 1
                    # Save base64 to instance variable for later use
                    self._current_images_base64.append(image_base64)

                    # Keep original path info and add VLM marker
                    result = f"Image Path: {image_path}\n[VLM_IMAGE_{images_processed}]"
                    self.logger.debug(
                        f"Successfully processed image {images_processed}: {image_path}"
                    )
                    return result
                else:
                    self.logger.error(f"Failed to encode image: {image_path}")
                    return match.group(0)  # Keep original if encoding failed

            except Exception as e:
                self.logger.error(f"Failed to process image {image_path}: {e}")
                return match.group(0)  # Keep original

        # Execute replacement
        enhanced_prompt = re.sub(
            image_path_pattern, replace_image_path, enhanced_prompt
        )

        return enhanced_prompt, images_processed

    def _build_vlm_messages_with_images(
        self, enhanced_prompt: str, user_query: str, system_prompt: str
    ) -> List[Dict]:
        """
        Build VLM message format, using markers to correspond images with text positions

        Args:
            enhanced_prompt: Enhanced prompt with image markers
            user_query: User query

        Returns:
            List[Dict]: VLM message format
        """
        images_base64 = getattr(self, "_current_images_base64", [])

        if not images_base64:
            # Pure text mode
            return [
                {
                    "role": "user",
                    "content": f"Context:\n{enhanced_prompt}\n\nUser Question: {user_query}",
                }
            ]

        # Build multimodal content
        content_parts = []

        # Split text at image markers and insert images
        text_parts = enhanced_prompt.split("[VLM_IMAGE_")

        for i, text_part in enumerate(text_parts):
            if i == 0:
                # First text part
                if text_part.strip():
                    content_parts.append({"type": "text", "text": text_part})
            else:
                # Find marker number and insert corresponding image
                marker_match = re.match(r"(\d+)\](.*)", text_part, re.DOTALL)
                if marker_match:
                    image_num = (
                        int(marker_match.group(1)) - 1
                    )  # Convert to 0-based index
                    remaining_text = marker_match.group(2)

                    # Insert corresponding image
                    if 0 <= image_num < len(images_base64):
                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{images_base64[image_num]}"
                                },
                            }
                        )

                    # Insert remaining text
                    if remaining_text.strip():
                        content_parts.append({"type": "text", "text": remaining_text})

        # Add user question
        content_parts.append(
            {
                "type": "text",
                "text": f"\n\nUser Question: {user_query}\n\nPlease answer based on the context and images provided.",
            }
        )
        base_system_prompt = "You are a helpful assistant that can analyze both text and image content to provide comprehensive answers."

        if system_prompt:
            full_system_prompt = base_system_prompt + " " + system_prompt
        else:
            full_system_prompt = base_system_prompt

        return [
            {
                "role": "system",
                "content": full_system_prompt,
            },
            {
                "role": "user",
                "content": content_parts,
            },
        ]

    async def _call_vlm_with_multimodal_content(self, messages: List[Dict]) -> str:
        """
        Call VLM to process multimodal content

        Args:
            messages: VLM message format

        Returns:
            str: VLM response result
        """
        try:
            user_message = messages[1]
            content = user_message["content"]
            system_prompt = messages[0]["content"]

            if isinstance(content, str):
                # Pure text mode
                result = await self.vision_model_func(
                    content, system_prompt=system_prompt
                )
            else:
                # Multimodal mode - pass complete messages directly to VLM
                result = await self.vision_model_func(
                    "",  # Empty prompt since we're using messages format
                    messages=messages,
                )

            return result

        except Exception as e:
            self.logger.error(f"VLM call failed: {e}")
            raise

    # Synchronous versions of query methods
    def query(self, query: str, mode: str = "mix", **kwargs) -> str:
        """
        Synchronous version of pure text query

        Args:
            query: Query text
            mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
            **kwargs: Other query parameters, will be passed to QueryParam
                - vlm_enhanced: bool, default True when vision_model_func is available.
                  If True, will parse image paths in retrieved context and replace them
                  with base64 encoded images for VLM processing.

        Returns:
            str: Query result
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, mode=mode, **kwargs))

    def query_with_multimodal(
        self,
        query: str,
        multimodal_content: List[Dict[str, Any]] = None,
        mode: str = "mix",
        **kwargs,
    ) -> str:
        """
        Synchronous version of multimodal query

        Args:
            query: Base query text
            multimodal_content: List of multimodal content, each element contains:
                - type: Content type ("image", "table", "equation", etc.)
                - Other fields depend on type (e.g., img_path, table_data, latex, etc.)
            mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
            **kwargs: Other query parameters, will be passed to QueryParam

        Returns:
            str: Query result
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_with_multimodal(query, multimodal_content, mode=mode, **kwargs)
        )

    async def get_all_documents(
        self, 
        filter_metadata: Dict[str, Any] = None,
        sort_by: str = "name"
    ) -> List[Dict[str, Any]]:
        """
        Get all processed documents with metadata
        
        Args:
            filter_metadata: Optional metadata filters (e.g., {"file_type": "pdf", "status": "PROCESSED"})
            sort_by: Sort field ("name", "date", "size")
            
        Returns:
            List of document dictionaries with metadata
        """
        documents = []
        
        try:
            if not hasattr(self.lightrag, "doc_status") or not self.lightrag.doc_status:
                self.logger.warning("doc_status storage not available")
                return []

            # Get all document IDs
            all_doc_ids = []
            if hasattr(self.lightrag.doc_status, "get_all_ids"):
                all_doc_ids = await self.lightrag.doc_status.get_all_ids()
            elif hasattr(self.lightrag.doc_status, "list_all"):
                all_doc_ids = await self.lightrag.doc_status.list_all()
            else:
                # Fallback: try to access underlying storage
                import json
                from pathlib import Path
                working_dir = Path(self.config.working_dir)
                doc_status_file = working_dir / "kv_store_doc_status.json"
                
                if doc_status_file.exists():
                    with open(doc_status_file, 'r') as f:
                        doc_status_data = json.load(f)
                        all_doc_ids = list(doc_status_data.keys()) if isinstance(doc_status_data, dict) else []

            # Get document details
            for doc_id in all_doc_ids:
                try:
                    doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
                    if not doc_status:
                        continue
                    
                    file_path = doc_status.get("file_path", "")
                    file_name = Path(file_path).name if file_path else doc_id
                    file_ext = Path(file_path).suffix.lower() if file_path else ""
                    
                    doc_info = {
                        "doc_id": doc_id,
                        "name": file_name,
                        "file_path": file_path,
                        "file_type": file_ext.lstrip('.') if file_ext else "unknown",
                        "status": doc_status.get("status", "UNKNOWN"),
                        "chunks_count": doc_status.get("chunks_count", 0),
                        "created_at": doc_status.get("created_at", ""),
                        "updated_at": doc_status.get("updated_at", ""),
                        "multimodal_processed": doc_status.get("multimodal_processed", False),
                    }
                    
                    # Apply metadata filters
                    if filter_metadata:
                        match = True
                        for key, value in filter_metadata.items():
                            if key not in doc_info or doc_info[key] != value:
                                match = False
                                break
                        if not match:
                            continue
                    
                    documents.append(doc_info)
                    
                except Exception as e:
                    self.logger.debug(f"Error getting document info for {doc_id}: {e}")
                    continue

            # Sort documents
            if sort_by == "name":
                documents.sort(key=lambda x: x.get("name", "").lower())
            elif sort_by == "date":
                documents.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            elif sort_by == "size":
                documents.sort(key=lambda x: x.get("chunks_count", 0), reverse=True)

        except Exception as e:
            self.logger.error(f"Error getting all documents: {e}")
        
        return documents

    async def _map_document_names_to_ids(
        self, document_names: List[str], return_metadata: bool = False
    ) -> Dict[str, Any]:
        """
        Map document names to document IDs by checking doc_status storage
        Enhanced with fuzzy matching and better partial name support

        Args:
            document_names: List of document names (file names or partial matches)
            return_metadata: If True, return metadata along with IDs

        Returns:
            Dict mapping document names to document IDs (or dicts with metadata if return_metadata=True)
        """
        if not document_names:
            return {}

        doc_name_to_id = {}
        
        try:
            # Get all document statuses
            if not hasattr(self.lightrag, "doc_status") or not self.lightrag.doc_status:
                self.logger.warning("doc_status storage not available")
                return {}

            # Try to get all document IDs from doc_status
            # LightRAG stores doc_status in KV storage
            all_doc_ids = []
            
            # Try different methods to get all doc IDs
            if hasattr(self.lightrag.doc_status, "get_all_ids"):
                all_doc_ids = await self.lightrag.doc_status.get_all_ids()
            elif hasattr(self.lightrag.doc_status, "list_all"):
                all_doc_ids = await self.lightrag.doc_status.list_all()
            else:
                # Fallback: try to access underlying storage
                # For file-based storage, scan the working directory
                import json
                from pathlib import Path
                working_dir = Path(self.config.working_dir)
                doc_status_file = working_dir / "kv_store_doc_status.json"
                
                if doc_status_file.exists():
                    with open(doc_status_file, 'r') as f:
                        doc_status_data = json.load(f)
                        all_doc_ids = list(doc_status_data.keys()) if isinstance(doc_status_data, dict) else []

            # Helper function for fuzzy matching
            def fuzzy_match(query: str, target: str) -> float:
                """Simple fuzzy matching score (0.0 to 1.0)"""
                query_lower = query.lower().strip()
                target_lower = target.lower().strip()
                
                # Exact match
                if query_lower == target_lower:
                    return 1.0
                
                # Substring match
                if query_lower in target_lower:
                    return 0.8
                if target_lower in query_lower:
                    return 0.7
                
                # Word-based matching
                query_words = set(query_lower.split())
                target_words = set(target_lower.split())
                if query_words and target_words:
                    common_words = query_words.intersection(target_words)
                    if common_words:
                        return len(common_words) / max(len(query_words), len(target_words))
                
                return 0.0

            # Match document names to IDs with fuzzy matching
            for doc_name in document_names:
                doc_name_lower = doc_name.lower().strip()
                # Normalize: remove underscores, convert to lowercase for better matching
                doc_name_normalized = doc_name_lower.replace("_", "").replace("-", "")
                best_match = None
                best_score = 0.0
                best_match_info = None
                
                for doc_id in all_doc_ids:
                    try:
                        doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
                        if not doc_status:
                            continue
                        
                        file_path = doc_status.get("file_path", "")
                        file_name = Path(file_path).name if file_path else ""
                        file_name_no_ext = Path(file_path).stem if file_path else ""
                        
                        # Calculate match scores for different fields
                        scores = []
                        match_details = []
                        
                        # Match against file name (normalized)
                        if file_name:
                            file_name_normalized = file_name.lower().replace("_", "").replace("-", "")
                            file_name_no_ext_normalized = file_name_no_ext.lower().replace("_", "").replace("-", "")
                            
                            # Check if query name is contained in file name (strong match)
                            if doc_name_normalized in file_name_normalized:
                                scores.append(0.9)
                                match_details.append(f"contains in filename")
                            if doc_name_normalized in file_name_no_ext_normalized:
                                scores.append(0.85)
                                match_details.append(f"contains in filename (no ext)")
                            
                            # Fuzzy match
                            scores.append(fuzzy_match(doc_name_lower, file_name.lower()))
                            scores.append(fuzzy_match(doc_name_normalized, file_name_normalized))
                        
                        # Match against doc_id
                        doc_id_normalized = doc_id.lower().replace("_", "").replace("-", "")
                        if doc_name_normalized in doc_id_normalized:
                            scores.append(0.8)
                            match_details.append(f"contains in doc_id")
                        scores.append(fuzzy_match(doc_name_lower, doc_id.lower()))
                        
                        # Match against file path components (check each part)
                        if file_path:
                            path_parts = Path(file_path).parts
                            for part in path_parts:
                                part_normalized = part.lower().replace("_", "").replace("-", "")
                                if doc_name_normalized in part_normalized:
                                    scores.append(0.7)
                                    match_details.append(f"contains in path part: {part}")
                                scores.append(fuzzy_match(doc_name_lower, part.lower()) * 0.5)
                        
                        max_score = max(scores) if scores else 0.0
                        
                        if max_score > best_score and max_score >= 0.2:  # Lower threshold for better matching
                            best_score = max_score
                            best_match_info = {
                                "file_name": file_name,
                                "file_path": file_path,
                                "match_details": match_details
                            }
                            if return_metadata:
                                best_match = {
                                    "doc_id": doc_id,
                                    "file_path": file_path,
                                    "file_name": file_name,
                                    "status": doc_status.get("status", ""),
                                    "chunks_count": doc_status.get("chunks_count", 0),
                                    "match_score": max_score,
                                }
                            else:
                                best_match = doc_id
                                
                    except Exception as e:
                        self.logger.debug(f"Error checking doc_id {doc_id}: {e}")
                        continue
                
                if best_match:
                    doc_name_to_id[doc_name] = best_match
                    match_info = f" (file: {best_match_info['file_name']})" if best_match_info else ""
                    self.logger.info(f"Matched '{doc_name}' to document (score: {best_score:.2f}){match_info}")
                else:
                    self.logger.warning(f"Could not find document ID for: {doc_name}")

        except Exception as e:
            self.logger.error(f"Error mapping document names to IDs: {e}")
        
        return doc_name_to_id

    async def _retrieve_chunks_with_document_filter(
        self,
        query: str,
        allowed_doc_ids: List[str] = None,
        top_k: int = 20,
        mode: str = "hybrid",
        filter_metadata: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval with document ID filtering and optional metadata filtering

        Args:
            query: Query text
            allowed_doc_ids: List of allowed document IDs (None means all documents)
            top_k: Number of chunks to retrieve
            mode: Retrieval mode ("hybrid", "local", "global")
            filter_metadata: Optional metadata filters (e.g., {"file_type": "pdf", "status": "PROCESSED"})

        Returns:
            List of retrieved chunks with metadata
        """
        # If no doc_ids specified, get all document IDs (unless metadata filter is applied)
        if allowed_doc_ids is None:
            if filter_metadata:
                # Apply metadata filter to get allowed doc IDs
                all_docs = await self.get_all_documents(filter_metadata=filter_metadata)
                allowed_doc_ids = [doc["doc_id"] for doc in all_docs]
            else:
                # Get all document IDs
                all_docs = await self.get_all_documents()
                allowed_doc_ids = [doc["doc_id"] for doc in all_docs]
        
        if not allowed_doc_ids:
            return []
        
        # Apply additional metadata filtering if specified
        if filter_metadata and allowed_doc_ids:
            filtered_doc_ids = []
            for doc_id in allowed_doc_ids:
                try:
                    doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
                    if not doc_status:
                        continue
                    
                    file_path = doc_status.get("file_path", "")
                    file_ext = Path(file_path).suffix.lower().lstrip('.') if file_path else ""
                    
                    doc_info = {
                        "file_type": file_ext or "unknown",
                        "status": doc_status.get("status", ""),
                    }
                    
                    # Check if document matches all metadata filters
                    match = True
                    for key, value in filter_metadata.items():
                        if key == "file_type" and doc_info.get("file_type") != value:
                            match = False
                            break
                        elif key == "status" and doc_info.get("status") != value:
                            match = False
                            break
                        elif key in doc_status and doc_status[key] != value:
                            match = False
                            break
                    
                    if match:
                        filtered_doc_ids.append(doc_id)
                except Exception as e:
                    self.logger.debug(f"Error checking metadata for doc_id {doc_id}: {e}")
                    continue
            
            allowed_doc_ids = filtered_doc_ids

        try:
            # Get query embedding
            query_embedding = await self.lightrag.embedding_func([query])
            
            # Check if embedding was generated (convert to list first to avoid numpy array boolean issues)
            if query_embedding is None:
                self.logger.error("Failed to generate query embedding: None returned")
                return []
            
            # Convert to list immediately to avoid numpy array boolean comparison issues
            if HAS_NUMPY and isinstance(query_embedding, (np.ndarray, list)):
                if isinstance(query_embedding, np.ndarray):
                    query_embedding = query_embedding.tolist()
                # If it's a list of arrays, convert each
                if len(query_embedding) > 0 and HAS_NUMPY and isinstance(query_embedding[0], np.ndarray):
                    query_embedding = [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in query_embedding]
            
            if len(query_embedding) == 0:
                self.logger.error("Failed to generate query embedding: Empty result")
                return []
            
            query_embedding = query_embedding[0]
            
            # Ensure it's a list (not numpy array) for all subsequent operations
            if HAS_NUMPY and isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Check embedding dimension matches stored vectors
            query_dim = len(query_embedding)
            try:
                # Try to get a stored vector to check dimension
                if hasattr(self.lightrag, "chunks_vdb") and self.lightrag.chunks_vdb:
                    # Get first chunk ID if possible
                    if hasattr(self.lightrag.text_chunks, "get_all_ids"):
                        chunk_ids = await self.lightrag.text_chunks.get_all_ids()
                        if chunk_ids:
                            # Try to get embedding for first chunk
                            chunk_data = await self.lightrag.text_chunks.get_by_id(chunk_ids[0])
                            if chunk_data and "embedding" in chunk_data:
                                stored_embedding = chunk_data["embedding"]
                                if isinstance(stored_embedding, list):
                                    stored_dim = len(stored_embedding)
                                    if stored_dim != query_dim:
                                        self.logger.error(
                                            f"CRITICAL: Embedding dimension mismatch! "
                                            f"Query embedding: {query_dim}D, Stored vectors: {stored_dim}D. "
                                            f"This will cause retrieval to fail. "
                                            f"Check EMBEDDING_DIM configuration."
                                        )
            except Exception as e:
                self.logger.debug(f"Could not verify embedding dimension: {e}")

            all_chunks = {}
            chunk_scores = {}

            # 1. Embedding-based retrieval with document filtering
            if mode in ["hybrid", "local", "global"]:
                try:
                    # Try different vector database search methods
                    embedding_results = []
                    
                    # Method 1: Try search method if available
                    if hasattr(self.lightrag.chunks_vdb, "search"):
                        try:
                            # Ensure query_embedding is a list (not numpy array) before search
                            search_embedding = query_embedding
                            if HAS_NUMPY and isinstance(query_embedding, np.ndarray):
                                search_embedding = query_embedding.tolist()
                            elif not isinstance(query_embedding, list):
                                search_embedding = list(query_embedding)
                            
                            # Search with larger top_k to ensure we get enough results after filtering
                            raw_results = await self.lightrag.chunks_vdb.search(
                                search_embedding, top_k=top_k * 5
                            )
                            
                            # Filter by document ID
                            for result in raw_results:
                                chunk_id = result.get("id") if isinstance(result, dict) else result
                                if not chunk_id:
                                    continue
                                
                                # Get chunk data from text_chunks (more reliable)
                                chunk_data = await self.lightrag.text_chunks.get_by_id(chunk_id)
                                if not chunk_data:
                                    # Try chunks_vdb as fallback
                                    chunk_data = await self.lightrag.chunks_vdb.get_by_id(chunk_id)
                                
                                if chunk_data:
                                    doc_id = chunk_data.get("full_doc_id", "")
                                    if doc_id in allowed_doc_ids:
                                        score = result.get("score", 0.0) if isinstance(result, dict) else 0.0
                                        embedding_results.append({
                                            "chunk_id": chunk_id,
                                            "chunk_data": chunk_data,
                                            "score": score,
                                        })
                        except Exception as e:
                            self.logger.debug(f"chunks_vdb.search() failed: {e}")
                    
                    # Method 2: Fallback - manual search through all chunks
                    if not embedding_results:
                        self.logger.info("Using fallback: manual chunk filtering")
                        # Get all chunk IDs and filter
                        all_chunk_ids = []
                        try:
                            # Try multiple methods to get chunk IDs
                            if hasattr(self.lightrag.text_chunks, "get_all_ids"):
                                all_chunk_ids = await self.lightrag.text_chunks.get_all_ids()
                            elif hasattr(self.lightrag.text_chunks, "list_all"):
                                all_chunk_ids = await self.lightrag.text_chunks.list_all()
                            elif hasattr(self.lightrag.text_chunks, "keys"):
                                # Try keys() method if available
                                all_chunk_ids = list(await self.lightrag.text_chunks.keys())
                            elif hasattr(self.lightrag.text_chunks, "_data"):
                                # Fallback: access internal data structure
                                all_chunk_ids = list(self.lightrag.text_chunks._data.keys())
                            
                            # If still empty, try to get from chunks_vdb
                            if not all_chunk_ids and hasattr(self.lightrag, "chunks_vdb"):
                                if hasattr(self.lightrag.chunks_vdb, "get_all_ids"):
                                    all_chunk_ids = await self.lightrag.chunks_vdb.get_all_ids()
                                elif hasattr(self.lightrag.chunks_vdb, "_data"):
                                    all_chunk_ids = list(self.lightrag.chunks_vdb._data.keys())
                                    
                        except Exception as e:
                            self.logger.warning(f"Could not get chunk IDs: {e}")
                            import traceback
                            self.logger.debug(f"Traceback: {traceback.format_exc()}")
                        
                        self.logger.info(f"Found {len(all_chunk_ids)} total chunks, filtering for {len(allowed_doc_ids)} documents")
                        
                        # Filter chunks by document ID and calculate similarity
                        processed_count = 0
                        for chunk_id in all_chunk_ids[:top_k * 20]:  # Limit for performance
                            try:
                                chunk_data = await self.lightrag.text_chunks.get_by_id(chunk_id)
                                if not chunk_data:
                                    continue
                                
                                doc_id = chunk_data.get("full_doc_id", "")
                                if doc_id not in allowed_doc_ids:
                                    continue
                                
                                processed_count += 1
                                
                                # Get chunk embedding if available
                                chunk_content = chunk_data.get("content", "")
                                if not chunk_content:
                                    self.logger.debug(f"Chunk {chunk_id} has no content, skipping")
                                    continue
                                
                                # Calculate cosine similarity manually
                                try:
                                    chunk_embedding = await self.lightrag.embedding_func([chunk_content])
                                    if chunk_embedding and len(chunk_embedding) > 0:
                                        chunk_emb = chunk_embedding[0]
                                        
                                        # Convert to list if numpy array to avoid comparison issues
                                        if HAS_NUMPY and isinstance(chunk_emb, np.ndarray):
                                            chunk_emb = chunk_emb.tolist()
                                        
                                        # Cosine similarity calculation
                                        if HAS_NUMPY:
                                            # Convert to numpy arrays for calculation
                                            query_arr = np.array(query_embedding)
                                            chunk_arr = np.array(chunk_emb)
                                            dot_product = float(np.dot(query_arr, chunk_arr))
                                            norm_query = float(np.linalg.norm(query_arr))
                                            norm_chunk = float(np.linalg.norm(chunk_arr))
                                        else:
                                            # Manual calculation without numpy
                                            dot_product = sum(a * b for a, b in zip(query_embedding, chunk_emb))
                                            norm_query = math.sqrt(sum(a * a for a in query_embedding))
                                            norm_chunk = math.sqrt(sum(a * a for a in chunk_emb))
                                        
                                        # Ensure norms are scalars (not arrays)
                                        norm_query = float(norm_query) if not isinstance(norm_query, (int, float)) else norm_query
                                        norm_chunk = float(norm_chunk) if not isinstance(norm_chunk, (int, float)) else norm_chunk
                                        
                                        if norm_query > 0 and norm_chunk > 0:
                                            similarity = dot_product / (norm_query * norm_chunk)
                                            # Accept all similarities, even low ones - let sorting handle it
                                            embedding_results.append({
                                                "chunk_id": chunk_id,
                                                "chunk_data": chunk_data,
                                                "score": float(similarity),
                                            })
                                            self.logger.debug(f"Added chunk {chunk_id} with similarity {similarity:.4f}")
                                except Exception as e:
                                    self.logger.warning(f"Error calculating similarity for chunk {chunk_id}: {e}")
                                    # Use default score if embedding fails - still include the chunk
                                    embedding_results.append({
                                        "chunk_id": chunk_id,
                                        "chunk_data": chunk_data,
                                        "score": 0.1,  # Low but non-zero score
                                    })
                            except Exception as e:
                                self.logger.debug(f"Error processing chunk {chunk_id}: {e}")
                                continue
                        
                        self.logger.info(f"Processed {processed_count} chunks, found {len(embedding_results)} with similarity scores")
                        
                        # Sort by score
                        embedding_results.sort(key=lambda x: x["score"], reverse=True)
                        embedding_results = embedding_results[:top_k * 2]
                        
                        if embedding_results:
                            top_scores = [f'{r["score"]:.4f}' for r in embedding_results[:5]]
                            self.logger.info(f"Top similarity scores: {top_scores}")
                    
                    # Add embedding results to all_chunks
                    for item in embedding_results:
                        chunk_id = item["chunk_id"]
                        all_chunks[chunk_id] = item["chunk_data"]
                        chunk_scores[chunk_id] = {
                            "embedding_score": item["score"],
                            "sources": ["embedding"]
                        }
                        
                except Exception as e:
                    self.logger.warning(f"Embedding retrieval failed: {e}")

            # 2. BM25-based retrieval with document filtering (for hybrid mode)
            if mode == "hybrid":
                try:
                    # Simple BM25-like keyword matching
                    query_terms = [t.lower() for t in query.split() if len(t) > 2]  # Filter short terms
                    
                    # Search through text chunks
                    all_chunk_ids = []
                    try:
                        if hasattr(self.lightrag.text_chunks, "get_all_ids"):
                            all_chunk_ids = await self.lightrag.text_chunks.get_all_ids()
                        elif hasattr(self.lightrag.text_chunks, "list_all"):
                            all_chunk_ids = await self.lightrag.text_chunks.list_all()
                    except Exception as e:
                        self.logger.debug(f"Could not get chunk IDs for BM25: {e}")
                    
                    bm25_results = []
                    for chunk_id in all_chunk_ids:
                        try:
                            chunk_data = await self.lightrag.text_chunks.get_by_id(chunk_id)
                            if not chunk_data:
                                continue
                            
                            doc_id = chunk_data.get("full_doc_id", "")
                            if doc_id not in allowed_doc_ids:
                                continue
                            
                            content = chunk_data.get("content", "").lower()
                            if not content:
                                continue
                            
                            # Simple term frequency scoring
                            score = 0.0
                            content_words = content.split()
                            total_words = len(content_words)
                            
                            for term in query_terms:
                                term_count = content.count(term)
                                if term_count > 0:
                                    # Simple TF-IDF-like scoring
                                    tf = term_count / max(total_words, 1)
                                    score += tf
                            
                            if score > 0:
                                bm25_results.append({
                                    "chunk_id": chunk_id,
                                    "chunk_data": chunk_data,
                                    "score": score,
                                })
                        except Exception as e:
                            self.logger.debug(f"Error in BM25 processing chunk {chunk_id}: {e}")
                            continue
                    
                    # Sort and limit BM25 results
                    bm25_results.sort(key=lambda x: x["score"], reverse=True)
                    bm25_results = bm25_results[:top_k * 2]
                    
                    # Add BM25 results to all_chunks
                    for item in bm25_results:
                        chunk_id = item["chunk_id"]
                        if chunk_id not in all_chunks:
                            all_chunks[chunk_id] = item["chunk_data"]
                            chunk_scores[chunk_id] = {
                                "bm25_score": item["score"],
                                "sources": ["bm25"]
                            }
                        else:
                            chunk_scores[chunk_id]["bm25_score"] = item["score"]
                            chunk_scores[chunk_id]["sources"].append("bm25")
                            
                except Exception as e:
                    self.logger.warning(f"BM25 retrieval failed: {e}")

            # 3. Combine and normalize scores
            combined_chunks = []
            for chunk_id, chunk_data in all_chunks.items():
                scores = chunk_scores.get(chunk_id, {})
                embedding_score = scores.get("embedding_score", 0.0)
                bm25_score = scores.get("bm25_score", 0.0)
                
                # Normalize scores
                if mode == "hybrid" and bm25_score > 0:
                    # Normalize BM25 score (assuming max score around 10)
                    normalized_bm25 = min(bm25_score / 10.0, 1.0)
                    # Combine embedding and BM25 scores
                    combined_score = (embedding_score * 0.7) + (normalized_bm25 * 0.3)
                else:
                    combined_score = embedding_score
                
                combined_chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_data": chunk_data,
                    "score": combined_score,
                    "sources": scores.get("sources", [])
                })

            # 4. Re-rank by combined score
            combined_chunks.sort(key=lambda x: x["score"], reverse=True)
            
            # Return top_k chunks
            return combined_chunks[:top_k]

        except Exception as e:
            self.logger.error(f"Error in document-filtered retrieval: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return []

    async def aquery_all_documents(
        self,
        query: str,
        mode: str = "hybrid",
        system_prompt: str | None = None,
        top_k: int = 20,
        filter_metadata: Dict[str, Any] = None,
        **kwargs,
    ) -> str:
        """
        Query across all documents without requiring explicit document names
        
        This function performs:
        1. Retrieval across all documents (or filtered by metadata)
        2. Groups results by document
        3. Returns unified answer with source attribution
        
        Args:
            query: Query text
            mode: Retrieval mode ("hybrid", "local", "global")
            system_prompt: Optional system prompt
            top_k: Number of chunks to retrieve
            filter_metadata: Optional metadata filters (e.g., {"file_type": "pdf"})
            **kwargs: Additional query parameters
            
        Returns:
            str: Query result with source attribution
        """
        if self.lightrag is None:
            raise ValueError(
                "No LightRAG instance available. Please process documents first."
            )
        
        # Get all document IDs (optionally filtered by metadata)
        all_docs = await self.get_all_documents(filter_metadata=filter_metadata)
        if not all_docs:
            self.logger.warning("No documents found in storage. Storage may be empty or path may be incorrect.")
            # Check if storage is actually empty
            try:
                if hasattr(self.lightrag, "doc_status") and self.lightrag.doc_status:
                    # Try to get all doc IDs directly
                    all_doc_ids = []
                    if hasattr(self.lightrag.doc_status, "get_all_ids"):
                        all_doc_ids = await self.lightrag.doc_status.get_all_ids()
                    elif hasattr(self.lightrag.doc_status, "list_all"):
                        all_doc_ids = await self.lightrag.doc_status.list_all()
                    
                    if not all_doc_ids:
                        self.logger.error(f"Storage appears to be empty. Working directory: {getattr(self.config, 'working_dir', 'unknown')}")
                        return "No documents found in storage. Please upload and process documents first."
            except Exception as e:
                self.logger.debug(f"Error checking storage: {e}")
            
            return "No documents found matching the criteria."
        
        allowed_doc_ids = [doc["doc_id"] for doc in all_docs]
        doc_id_to_name = {doc["doc_id"]: doc["name"] for doc in all_docs}
        
        self.logger.info(
            f"Querying across {len(allowed_doc_ids)} documents"
        )
        
        # Retrieve chunks across all documents
        retrieved_chunks = await self._retrieve_chunks_with_document_filter(
            query=query,
            allowed_doc_ids=allowed_doc_ids,
            top_k=top_k * len(allowed_doc_ids),  # Get more chunks to ensure coverage
            mode=mode,
            filter_metadata=filter_metadata,
        )
        
        if not retrieved_chunks:
            self.logger.warning(f"No relevant chunks retrieved for query: {query[:100]}...")
            self.logger.warning(f"Query was executed across {len(allowed_doc_ids)} documents but no matching content was found.")
            return "No relevant content found in the documents."
        
        # Group chunks by document
        chunks_by_doc = {}
        for chunk_item in retrieved_chunks:
            chunk_data = chunk_item["chunk_data"]
            doc_id = chunk_data.get("full_doc_id", "")
            
            if doc_id in doc_id_to_name:
                doc_name = doc_id_to_name[doc_id]
                if doc_name not in chunks_by_doc:
                    chunks_by_doc[doc_name] = []
                chunks_by_doc[doc_name].append(chunk_item)
        
        # Build context grouped by document with source attribution
        context_parts = []
        for doc_name, chunks in chunks_by_doc.items():
            context_parts.append(f"\n\n[Document: {doc_name}]\n")
            context_parts.append("-" * 80 + "\n")
            
            for chunk_item in chunks:
                chunk_data = chunk_item["chunk_data"]
                content = chunk_data.get("content", "")
                score = chunk_item.get("score", 0.0)
                if content:
                    context_parts.append(f"{content}\n\n")
        
        combined_context = "".join(context_parts)
        
        # Build prompt with source attribution
        query_prompt = f"""Based on the following content from multiple documents, answer the user's question accurately and concisely.

User Query: {query}

Content from Documents:
{combined_context}

Please provide a comprehensive answer based on the source material. For each piece of information, cite which document it comes from using the format [Document: name]. Focus on:
1. Direct information from the documents
2. Clear source attribution for each fact
3. Cross-document comparisons when relevant
4. Any contradictions or complementary information

Use the actual text from the documents rather than generating summaries. Always cite your sources."""
        
        # Make LLM call
        try:
            result = await self.lightrag.llm_model_func(
                query_prompt,
                system_prompt=system_prompt,
                history_messages=[],
                **kwargs,
            )
            
            self.logger.info("All-documents query completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in all-documents LLM call: {e}")
            raise

    async def aquery_multi_document(
        self,
        query: str,
        document_names: List[str] = None,
        mode: str = "hybrid",
        query_mode: str = "auto",
        system_prompt: str | None = None,
        top_k: int = 20,
        filter_metadata: Dict[str, Any] = None,
        **kwargs,
    ) -> str:
        """
        Multi-document comparison query using single hybrid retrieval + single LLM call
        Enhanced with flexible query modes: "all", "selected", "auto"

        This function performs:
        1. Single hybrid retrieval across selected/all documents
        2. Filters chunks by document IDs or metadata
        3. Combines BM25 and embedding results
        4. Applies cross-document re-ranking
        5. Groups chunks by document
        6. Makes exactly ONE LLM call with all relevant chunks

        Args:
            query: Query text
            document_names: List of document names to compare (optional, depends on query_mode)
            mode: Retrieval mode ("hybrid", "local", "global")
            query_mode: Query mode - "all" (all documents), "selected" (specified documents), "auto" (detect or use all)
            system_prompt: Optional system prompt
            top_k: Number of chunks to retrieve per document
            filter_metadata: Optional metadata filters (e.g., {"file_type": "pdf"})
            **kwargs: Additional query parameters

        Returns:
            str: Comparison result from single LLM call
        """
        if self.lightrag is None:
            raise ValueError(
                "No LightRAG instance available. Please process documents first."
            )

        # Determine which documents to query based on query_mode
        doc_name_to_id = None
        allowed_doc_ids = None
        
        if query_mode == "all":
            # Query all documents
            return await self.aquery_all_documents(
                query=query,
                mode=mode,
                system_prompt=system_prompt,
                top_k=top_k,
                filter_metadata=filter_metadata,
                **kwargs,
            )
        elif query_mode == "selected":
            # Query only specified documents
            if not document_names:
                raise ValueError(
                    "document_names must be provided when query_mode is 'selected'"
                )
            
            doc_name_to_id = await self._map_document_names_to_ids(document_names)
            
            if not doc_name_to_id:
                raise ValueError(
                    f"Could not find any documents matching: {document_names}"
                )

            allowed_doc_ids = list(doc_name_to_id.values())
            self.logger.info(
                f"Found {len(allowed_doc_ids)} documents for comparison: {list(doc_name_to_id.keys())}"
            )
        elif query_mode == "auto":
            # Auto-detect: use document_names if provided, otherwise use all
            if document_names and len(document_names) > 0:
                doc_name_to_id = await self._map_document_names_to_ids(document_names)
                
                if doc_name_to_id:
                    allowed_doc_ids = list(doc_name_to_id.values())
                    self.logger.info(
                        f"Auto-mode: Found {len(allowed_doc_ids)} documents: {list(doc_name_to_id.keys())}"
                    )
                else:
                    # Fallback to all documents if no matches
                    self.logger.info("Auto-mode: No document matches found, querying all documents")
                    return await self.aquery_all_documents(
                        query=query,
                        mode=mode,
                        system_prompt=system_prompt,
                        top_k=top_k,
                        filter_metadata=filter_metadata,
                        **kwargs,
                    )
            else:
                # No document names provided, query all
                self.logger.info("Auto-mode: No document names provided, querying all documents")
                return await self.aquery_all_documents(
                    query=query,
                    mode=mode,
                    system_prompt=system_prompt,
                    top_k=top_k,
                    filter_metadata=filter_metadata,
                    **kwargs,
                )
        else:
            raise ValueError(f"Invalid query_mode: {query_mode}. Must be 'all', 'selected', or 'auto'")

        # Get document names mapping for selected documents
        if doc_name_to_id:
            doc_id_to_name = {v: k for k, v in doc_name_to_id.items()}
        else:
            raise ValueError("Failed to determine document mapping")

        # Step 2: Single hybrid retrieval across selected documents
        retrieved_chunks = await self._retrieve_chunks_with_document_filter(
            query=query,
            allowed_doc_ids=allowed_doc_ids,
            top_k=top_k * len(allowed_doc_ids),  # Get more chunks to ensure coverage
            mode=mode,
            filter_metadata=filter_metadata,
        )

        if not retrieved_chunks:
            return "No relevant content found in the specified documents."

        # Step 3: Group chunks by document
        chunks_by_doc = {}
        for chunk_item in retrieved_chunks:
            chunk_data = chunk_item["chunk_data"]
            doc_id = chunk_data.get("full_doc_id", "")
            
            if doc_id in doc_id_to_name:
                doc_name = doc_id_to_name[doc_id]
                if doc_name not in chunks_by_doc:
                    chunks_by_doc[doc_name] = []
                chunks_by_doc[doc_name].append(chunk_item)

        # Step 4: Build context grouped by document with enhanced attribution
        context_parts = []
        for doc_name in doc_id_to_name.values():
            if doc_name in chunks_by_doc:
                context_parts.append(f"\n\n[Document: {doc_name}]\n")
                context_parts.append("-" * 80 + "\n")
                
                for chunk_item in chunks_by_doc[doc_name]:
                    chunk_data = chunk_item["chunk_data"]
                    content = chunk_data.get("content", "")
                    score = chunk_item.get("score", 0.0)
                    if content:
                        context_parts.append(f"{content}\n\n")

        combined_context = "".join(context_parts)

        # Step 5: Build prompt for single LLM call with enhanced source attribution
        comparison_prompt = f"""Based on the following content from multiple documents, please compare and analyze them according to the user's query.

User Query: {query}

Content from Documents:
{combined_context}

Please provide a comprehensive comparison based on the original source text from the documents. Focus on:
1. Key similarities and differences
2. Specific details from each document
3. Direct quotes and citations from the source material
4. Any contradictions or complementary information

Use the actual text from the documents rather than generating summaries. Always cite which document each piece of information comes from using the format [Document: name]."""

        # Step 6: Make exactly ONE LLM call
        try:
            # Use LightRAG's LLM function directly
            result = await self.lightrag.llm_model_func(
                comparison_prompt,
                system_prompt=system_prompt,
                history_messages=[],
                **kwargs,
            )
            
            self.logger.info("Multi-document comparison query completed")
            return result

        except Exception as e:
            self.logger.error(f"Error in multi-document LLM call: {e}")
            raise

    def query_multi_document(
        self,
        query: str,
        document_names: List[str] = None,
        mode: str = "hybrid",
        query_mode: str = "auto",
        system_prompt: str | None = None,
        top_k: int = 20,
        filter_metadata: Dict[str, Any] = None,
        **kwargs,
    ) -> str:
        """
        Synchronous version of multi-document comparison query

        Args:
            query: Query text
            document_names: List of document names to compare
            mode: Retrieval mode ("hybrid", "local", "global")
            query_mode: Query mode ("all", "selected", "auto")
            system_prompt: Optional system prompt
            top_k: Number of chunks to retrieve per document
            filter_metadata: Optional metadata filters
            **kwargs: Additional query parameters

        Returns:
            str: Comparison result from single LLM call
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_multi_document(
                query, document_names, mode, query_mode, system_prompt, top_k, filter_metadata, **kwargs
            )
        )
    
    def query_all_documents(
        self,
        query: str,
        mode: str = "hybrid",
        system_prompt: str | None = None,
        top_k: int = 20,
        filter_metadata: Dict[str, Any] = None,
        **kwargs,
    ) -> str:
        """
        Synchronous version of all-documents query
        
        Args:
            query: Query text
            mode: Retrieval mode ("hybrid", "local", "global")
            system_prompt: Optional system prompt
            top_k: Number of chunks to retrieve
            filter_metadata: Optional metadata filters
            **kwargs: Additional query parameters
            
        Returns:
            str: Query result with source attribution
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_all_documents(
                query, mode, system_prompt, top_k, filter_metadata, **kwargs
            )
        )
