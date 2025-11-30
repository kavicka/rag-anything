"""
Utility functions for RAGAnything

Contains helper functions for content separation, text insertion, and other utilities
"""

import base64
from typing import Dict, List, Any, Tuple
from pathlib import Path
from lightrag.utils import logger


def separate_content(
    content_list: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Separate text content and multimodal content

    Args:
        content_list: Content list from MinerU parsing

    Returns:
        (text_content, multimodal_items): Pure text content and multimodal items list
    """
    text_parts = []
    multimodal_items = []

    for item in content_list:
        content_type = item.get("type", "text")

        if content_type == "text":
            # Text content
            text = item.get("text", "")
            if text.strip():
                text_parts.append(text)
        else:
            # Multimodal content (image, table, equation, etc.)
            multimodal_items.append(item)

    # Merge all text content
    text_content = "\n\n".join(text_parts)

    logger.info("Content separation complete:")
    logger.info(f"  - Text content length: {len(text_content)} characters")
    logger.info(f"  - Multimodal items count: {len(multimodal_items)}")

    # Count multimodal types
    modal_types = {}
    for item in multimodal_items:
        modal_type = item.get("type", "unknown")
        modal_types[modal_type] = modal_types.get(modal_type, 0) + 1

    if modal_types:
        logger.info(f"  - Multimodal type distribution: {modal_types}")

    return text_content, multimodal_items


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 string

    Args:
        image_path: Path to the image file

    Returns:
        str: Base64 encoded string, empty string if encoding fails
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return ""


def resolve_image_path(image_path: str, output_dir: str = None) -> str:
    """
    Resolve image path, trying multiple strategies if the original path doesn't exist
    
    Args:
        image_path: Original image path (can be absolute or relative)
        output_dir: Output directory to search in if path doesn't exist
        
    Returns:
        str: Resolved image path, or original path if not found
    """
    try:
        path = Path(image_path)
        
        # If path exists, return it
        if path.exists():
            return str(path.resolve())
        
        # If path is absolute but doesn't exist, try to find it relative to output_dir
        if path.is_absolute() and output_dir:
            # Extract just the filename
            filename = path.name
            
            # Try to find it in common locations within output_dir
            # Pattern: output_dir/{uuid}/auto/images/filename or output_dir/{uuid}/images/filename
            output_dir_path = Path(output_dir)
            if output_dir_path.exists():
                # First, try searching in subdirectories that match the pattern
                # Look for images directories in any subdirectory
                for subdir in output_dir_path.iterdir():
                    if subdir.is_dir():
                        # Try auto/images/ first (most common structure)
                        auto_images_path = subdir / "auto" / "images" / filename
                        if auto_images_path.exists():
                            logger.info(f"Resolved image path: {image_path} -> {auto_images_path}")
                            return str(auto_images_path.resolve())
                        
                        # Try images/ directly
                        images_path = subdir / "images" / filename
                        if images_path.exists():
                            logger.info(f"Resolved image path: {image_path} -> {images_path}")
                            return str(images_path.resolve())
                
                # Fallback: recursive search for the filename
                for img_file in output_dir_path.rglob(filename):
                    if img_file.is_file():
                        logger.info(f"Resolved image path: {image_path} -> {img_file}")
                        return str(img_file.resolve())
        
        # If relative path, try resolving relative to output_dir
        if not path.is_absolute() and output_dir:
            output_path = Path(output_dir) / image_path
            if output_path.exists():
                logger.info(f"Resolved relative image path: {image_path} -> {output_path}")
                return str(output_path.resolve())
        
        # Return original path if we can't resolve it
        return image_path
        
    except Exception as e:
        logger.error(f"Error resolving image path {image_path}: {e}")
        return image_path


def validate_image_file(image_path: str, max_size_mb: int = 50, output_dir: str = None) -> bool:
    """
    Validate if a file is a valid image file

    Args:
        image_path: Path to the image file
        max_size_mb: Maximum file size in MB
        output_dir: Optional output directory to search in if path doesn't exist

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Try to resolve the path first
        resolved_path = resolve_image_path(image_path, output_dir)
        path = Path(resolved_path)

        logger.debug(f"Validating image path: {image_path} (resolved: {resolved_path})")
        logger.debug(f"Resolved path object: {path}")
        logger.debug(f"Path exists check: {path.exists()}")

        # Check if file exists
        if not path.exists():
            if resolved_path != image_path:
                logger.warning(f"Image file not found at original path: {image_path}, tried resolved path: {resolved_path}")
            else:
                logger.warning(f"Image file not found: {image_path}")
            return False

        # Check file extension
        image_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
            ".tiff",
            ".tif",
        ]

        path_lower = str(path).lower()
        has_valid_extension = any(path_lower.endswith(ext) for ext in image_extensions)
        logger.debug(
            f"File extension check - path: {path_lower}, valid: {has_valid_extension}"
        )

        if not has_valid_extension:
            logger.warning(f"File does not appear to be an image: {image_path}")
            return False

        # Check file size
        file_size = path.stat().st_size
        max_size = max_size_mb * 1024 * 1024
        logger.debug(
            f"File size check - size: {file_size} bytes, max: {max_size} bytes"
        )

        if file_size > max_size:
            logger.warning(f"Image file too large ({file_size} bytes): {image_path}")
            return False

        logger.debug(f"Image validation successful: {image_path}")
        return True

    except Exception as e:
        logger.error(f"Error validating image file {image_path}: {e}")
        return False


async def insert_text_content(
    lightrag,
    input: str | list[str],
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
):
    """
    Insert pure text content into LightRAG

    Args:
        lightrag: LightRAG instance
        input: Single document string or list of document strings
        split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
        chunk_token_size, it will be split again by token size.
        split_by_character_only: if split_by_character_only is True, split the string by character only, when
        split_by_character is None, this parameter is ignored.
        ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
        file_paths: single string of the file path or list of file paths, used for citation
    """
    logger.info("Starting text content insertion into LightRAG...")

    # Use LightRAG's insert method with all parameters
    await lightrag.ainsert(
        input=input,
        file_paths=file_paths,
        split_by_character=split_by_character,
        split_by_character_only=split_by_character_only,
        ids=ids,
    )

    logger.info("Text content insertion complete")


async def insert_text_content_with_multimodal_content(
    lightrag,
    input: str | list[str],
    multimodal_content: list[dict[str, any]] | None = None,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
    scheme_name: str | None = None,
):
    """
    Insert pure text content into LightRAG

    Args:
        lightrag: LightRAG instance
        input: Single document string or list of document strings
        multimodal_content: Multimodal content list (optional)
        split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
        chunk_token_size, it will be split again by token size.
        split_by_character_only: if split_by_character_only is True, split the string by character only, when
        split_by_character is None, this parameter is ignored.
        ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
        file_paths: single string of the file path or list of file paths, used for citation
        scheme_name: scheme name (optional)
    """
    logger.info("Starting text content insertion into LightRAG...")

    # Use LightRAG's insert method with all parameters
    try:
        await lightrag.ainsert(
            input=input,
            multimodal_content=multimodal_content,
            file_paths=file_paths,
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
            ids=ids,
            scheme_name=scheme_name,
        )
    except Exception as e:
        logger.info(f"Error: {e}")
        logger.info(
            "If the error is caused by the ainsert function not having a multimodal content parameter, please update the raganything branch of lightrag"
        )

    logger.info("Text content insertion complete")


def get_processor_for_type(modal_processors: Dict[str, Any], content_type: str):
    """
    Get appropriate processor based on content type

    Args:
        modal_processors: Dictionary of available processors
        content_type: Content type

    Returns:
        Corresponding processor instance
    """
    # Direct mapping to corresponding processor
    if content_type == "image":
        return modal_processors.get("image")
    elif content_type == "table":
        return modal_processors.get("table")
    elif content_type == "equation":
        return modal_processors.get("equation")
    else:
        # For other types, use generic processor
        return modal_processors.get("generic")


def get_processor_supports(proc_type: str) -> List[str]:
    """Get processor supported features"""
    supports_map = {
        "image": [
            "Image content analysis",
            "Visual understanding",
            "Image description generation",
            "Image entity extraction",
        ],
        "table": [
            "Table structure analysis",
            "Data statistics",
            "Trend identification",
            "Table entity extraction",
        ],
        "equation": [
            "Mathematical formula parsing",
            "Variable identification",
            "Formula meaning explanation",
            "Formula entity extraction",
        ],
        "generic": [
            "General content analysis",
            "Structured processing",
            "Entity extraction",
        ],
    }
    return supports_map.get(proc_type, ["Basic processing"])
