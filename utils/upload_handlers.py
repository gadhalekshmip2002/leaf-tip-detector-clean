# utils/upload_handlers.py
import streamlit as st
import tempfile
import os
import zipfile
from pathlib import Path
from PIL import Image
from config.supabase_storage import supabase_storage

def handle_single_image_upload(uploaded_file, source_page: str):
    """
    Unified single image upload handler for all tabs
    
    Args:
        uploaded_file: Streamlit uploaded file
        source_page: Which page/tab uploaded this (quick_detection, entire_image, etc.)
    
    Returns:
        tuple: (local_temp_path, storage_info)
    """
    try:
        # Create local temp file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        
        # Reset file pointer for Supabase upload
        uploaded_file.seek(0)
        
        # Upload to Supabase storage
        storage_info = None
        if supabase_storage.is_connected():
            storage_info = supabase_storage.upload_image(uploaded_file, source_page)
        
        return temp_path, storage_info
        
    except Exception as e:
        st.error(f"❌ Error handling image upload: {str(e)}")
        return None, None

def handle_batch_folder_upload(uploaded_folder, source_page: str):
    """
    Unified batch folder upload handler for all tabs
    
    Args:
        uploaded_folder: Streamlit uploaded ZIP file
        source_page: Which page/tab uploaded this
    
    Returns:
        tuple: (local_paths, original_names, storage_info_list)
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract ZIP file
            zip_path = os.path.join(temp_dir, "upload.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_folder.read())
            
            extract_dir = os.path.join(temp_dir, "extracted")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find all image files
            image_files = []
            original_names = []
            seen_names = set()
            
            # Look for common image extensions
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                for img_file in Path(extract_dir).rglob(f"*{ext}"):
                    if img_file.name not in seen_names:
                        image_files.append(img_file)
                        original_names.append(img_file.name)
                        seen_names.add(img_file.name)
                for img_file in Path(extract_dir).rglob(f"*{ext.upper()}"):
                    if img_file.name not in seen_names:
                        image_files.append(img_file)
                        original_names.append(img_file.name)
                        seen_names.add(img_file.name)
            
            if not image_files:
                st.warning("⚠️ No image files found in ZIP archive")
                return [], [], []
            
            # Create persistent local files + prepare for Supabase
            persistent_files = []
            batch_data_for_supabase = []
            
            for img_file in image_files:
                # Create persistent local file for processing
                with tempfile.NamedTemporaryFile(delete=False, suffix=img_file.suffix) as tmp:
                    with open(img_file, 'rb') as src:
                        file_content = src.read()
                        tmp.write(file_content)
                    persistent_files.append(tmp.name)
                    
                    # Prepare for Supabase batch upload
                    batch_data_for_supabase.append((file_content, img_file.name))
            
            # Upload batch to Supabase
            storage_info_list = []
            if supabase_storage.is_connected():
                storage_info_list = supabase_storage.upload_batch(batch_data_for_supabase, source_page)
            
            return persistent_files, original_names, storage_info_list
            
    except Exception as e:
        st.error(f"❌ Error handling batch upload: {str(e)}")
        return [], [], []

def show_storage_status_sidebar():
    """Show Supabase storage status in sidebar"""
    with st.sidebar:
        st.markdown("### 📦 Storage Status")
        if supabase_storage.is_connected():
            st.success("✅ Supabase Connected")
            st.caption("Images are being stored to cloud")
            
            # Show session info
            session_id = supabase_storage.get_session_id()
            st.caption(f"Session: {session_id[:8]}...")
        else:
            st.warning("⚠️ Storage Offline")
            st.caption("Images stored locally only")

def get_user_session_id():
    """Get current user session ID"""
    return supabase_storage.get_session_id()