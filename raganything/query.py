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

    async def _map_document_names_to_ids(
        self, document_names: List[str]
    ) -> Dict[str, str]:
        """
        Map document names to document IDs by checking doc_status storage

        Args:
            document_names: List of document names (file names or partial matches)

        Returns:
            Dict mapping document names to document IDs
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

            # Match document names to IDs
            for doc_name in document_names:
                doc_name_lower = doc_name.lower()
                matched = False
                
                for doc_id in all_doc_ids:
                    try:
                        doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
                        if doc_status:
                            file_path = doc_status.get("file_path", "")
                            file_name = Path(file_path).name if file_path else ""
                            
                            # Try exact match or partial match
                            if (
                                doc_name_lower in file_name.lower()
                                or file_name.lower() in doc_name_lower
                                or doc_name_lower in doc_id.lower()
                            ):
                                doc_name_to_id[doc_name] = doc_id
                                matched = True
                                break
                    except Exception as e:
                        self.logger.debug(f"Error checking doc_id {doc_id}: {e}")
                        continue
                
                if not matched:
                    self.logger.warning(f"Could not find document ID for: {doc_name}")

        except Exception as e:
            self.logger.error(f"Error mapping document names to IDs: {e}")
        
        return doc_name_to_id

    async def _retrieve_chunks_with_document_filter(
        self,
        query: str,
        allowed_doc_ids: List[str],
        top_k: int = 20,
        mode: str = "hybrid",
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval with document ID filtering

        Args:
            query: Query text
            allowed_doc_ids: List of allowed document IDs
            top_k: Number of chunks to retrieve
            mode: Retrieval mode ("hybrid", "local", "global")

        Returns:
            List of retrieved chunks with metadata
        """
        if not allowed_doc_ids:
            return []

        try:
            # Get query embedding
            query_embedding = await self.lightrag.embedding_func([query])
            if not query_embedding or len(query_embedding) == 0:
                self.logger.error("Failed to generate query embedding")
                return []
            query_embedding = query_embedding[0]

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
                            # Search with larger top_k to ensure we get enough results after filtering
                            raw_results = await self.lightrag.chunks_vdb.search(
                                query_embedding, top_k=top_k * 5
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
                            if hasattr(self.lightrag.text_chunks, "get_all_ids"):
                                all_chunk_ids = await self.lightrag.text_chunks.get_all_ids()
                            elif hasattr(self.lightrag.text_chunks, "list_all"):
                                all_chunk_ids = await self.lightrag.text_chunks.list_all()
                        except Exception as e:
                            self.logger.warning(f"Could not get chunk IDs: {e}")
                        
                        # Filter chunks by document ID and calculate similarity
                        for chunk_id in all_chunk_ids[:top_k * 20]:  # Limit for performance
                            try:
                                chunk_data = await self.lightrag.text_chunks.get_by_id(chunk_id)
                                if not chunk_data:
                                    continue
                                
                                doc_id = chunk_data.get("full_doc_id", "")
                                if doc_id not in allowed_doc_ids:
                                    continue
                                
                                # Get chunk embedding if available
                                chunk_content = chunk_data.get("content", "")
                                if not chunk_content:
                                    continue
                                
                                # Calculate cosine similarity manually
                                try:
                                    chunk_embedding = await self.lightrag.embedding_func([chunk_content])
                                    if chunk_embedding and len(chunk_embedding) > 0:
                                        chunk_emb = chunk_embedding[0]
                                        # Cosine similarity calculation
                                        if HAS_NUMPY:
                                            dot_product = np.dot(query_embedding, chunk_emb)
                                            norm_query = np.linalg.norm(query_embedding)
                                            norm_chunk = np.linalg.norm(chunk_emb)
                                        else:
                                            # Manual calculation without numpy
                                            dot_product = sum(a * b for a, b in zip(query_embedding, chunk_emb))
                                            norm_query = math.sqrt(sum(a * a for a in query_embedding))
                                            norm_chunk = math.sqrt(sum(a * a for a in chunk_emb))
                                        
                                        if norm_query > 0 and norm_chunk > 0:
                                            similarity = dot_product / (norm_query * norm_chunk)
                                            embedding_results.append({
                                                "chunk_id": chunk_id,
                                                "chunk_data": chunk_data,
                                                "score": float(similarity),
                                            })
                                except Exception as e:
                                    self.logger.debug(f"Error calculating similarity for chunk {chunk_id}: {e}")
                                    # Use default score if embedding fails
                                    embedding_results.append({
                                        "chunk_id": chunk_id,
                                        "chunk_data": chunk_data,
                                        "score": 0.5,
                                    })
                            except Exception as e:
                                self.logger.debug(f"Error processing chunk {chunk_id}: {e}")
                                continue
                        
                        # Sort by score
                        embedding_results.sort(key=lambda x: x["score"], reverse=True)
                        embedding_results = embedding_results[:top_k * 2]
                    
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
            self.logger.debug(traceback.format_exc())
            return []

    async def aquery_multi_document(
        self,
        query: str,
        document_names: List[str] = None,
        mode: str = "hybrid",
        system_prompt: str | None = None,
        top_k: int = 20,
        **kwargs,
    ) -> str:
        """
        Multi-document comparison query using single hybrid retrieval + single LLM call

        This function performs:
        1. Single hybrid retrieval across all selected documents
        2. Filters chunks by document IDs
        3. Combines BM25 and embedding results
        4. Applies cross-document re-ranking
        5. Groups chunks by document
        6. Makes exactly ONE LLM call with all relevant chunks

        Args:
            query: Query text
            document_names: List of document names to compare (if None, will try to detect from query)
            mode: Retrieval mode ("hybrid", "local", "global")
            system_prompt: Optional system prompt
            top_k: Number of chunks to retrieve per document
            **kwargs: Additional query parameters

        Returns:
            str: Comparison result from single LLM call
        """
        if self.lightrag is None:
            raise ValueError(
                "No LightRAG instance available. Please process documents first."
            )

        # Step 1: Map document names to document IDs
        if not document_names:
            # Try to detect document names from query (simple heuristic)
            # This is a basic implementation - can be enhanced
            self.logger.info("No document names provided, attempting to detect from query")
            # For now, return error - user should provide document names
            raise ValueError(
                "document_names must be provided for multi-document comparison"
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

        # Step 2: Single hybrid retrieval across all selected documents
        retrieved_chunks = await self._retrieve_chunks_with_document_filter(
            query=query,
            allowed_doc_ids=allowed_doc_ids,
            top_k=top_k * len(allowed_doc_ids),  # Get more chunks to ensure coverage
            mode=mode,
        )

        if not retrieved_chunks:
            return "No relevant content found in the specified documents."

        # Step 3: Group chunks by document
        chunks_by_doc = {}
        doc_id_to_name = {v: k for k, v in doc_name_to_id.items()}
        
        for chunk_item in retrieved_chunks:
            chunk_data = chunk_item["chunk_data"]
            doc_id = chunk_data.get("full_doc_id", "")
            
            if doc_id in doc_id_to_name:
                doc_name = doc_id_to_name[doc_id]
                if doc_name not in chunks_by_doc:
                    chunks_by_doc[doc_name] = []
                chunks_by_doc[doc_name].append(chunk_item)

        # Step 4: Build context grouped by document
        context_parts = []
        for doc_name in doc_name_to_id.keys():
            if doc_name in chunks_by_doc:
                context_parts.append(f"\n\nDocument: {doc_name}\n")
                context_parts.append("-" * 80 + "\n")
                
                for chunk_item in chunks_by_doc[doc_name]:
                    chunk_data = chunk_item["chunk_data"]
                    content = chunk_data.get("content", "")
                    if content:
                        context_parts.append(f"{content}\n\n")

        combined_context = "".join(context_parts)

        # Step 5: Build prompt for single LLM call
        comparison_prompt = f"""Based on the following content from multiple documents, please compare and analyze them according to the user's query.

User Query: {query}

Content from Documents:
{combined_context}

Please provide a comprehensive comparison based on the original source text from the documents. Focus on:
1. Key similarities and differences
2. Specific details from each document
3. Direct quotes and citations from the source material
4. Any contradictions or complementary information

Use the actual text from the documents rather than generating summaries. Cite which document each piece of information comes from."""

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
        system_prompt: str | None = None,
        top_k: int = 20,
        **kwargs,
    ) -> str:
        """
        Synchronous version of multi-document comparison query

        Args:
            query: Query text
            document_names: List of document names to compare
            mode: Retrieval mode ("hybrid", "local", "global")
            system_prompt: Optional system prompt
            top_k: Number of chunks to retrieve per document
            **kwargs: Additional query parameters

        Returns:
            str: Comparison result from single LLM call
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_multi_document(
                query, document_names, mode, system_prompt, top_k, **kwargs
            )
        )
