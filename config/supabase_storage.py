# config/supabase_storage.py
import os
import streamlit as st
from dotenv import load_dotenv 
from supabase import create_client, Client
from typing import Optional, Dict
import uuid
from datetime import datetime
from PIL import Image

class SupabaseImageStorage:
    def __init__(self):
        load_dotenv()
        # Get credentials from Streamlit secrets (deployment) or environment (local)
        try:
            # For Streamlit Cloud deployment
            self.supabase_url = st.secrets["SUPABASE_URL"]
            self.supabase_key = st.secrets["SUPABASE_ANON_KEY"]
        except:
            # For local testing with .env file
            self.supabase_url = os.getenv("SUPABASE_URL")
            self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        self.client: Optional[Client] = None
        self.bucket_name = "leaf-tip-images"
        
        if self.supabase_url and self.supabase_key:
            try:
                self.client = create_client(self.supabase_url, self.supabase_key)
                # Only show success message in local testing
                if os.getenv("SUPABASE_URL"):  # Local testing
                    print("✅ Supabase connected successfully!")
            except Exception as e:
                st.error(f"❌ Supabase connection failed: {str(e)}")
        else:
            st.warning("⚠️ Supabase credentials not found. Storage disabled.")
    
    def is_connected(self) -> bool:
        return self.client is not None
    
    def get_session_id(self) -> str:
        """Get or create session ID"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        return st.session_state.session_id
    
    def upload_image(self, image_file, source_page: str) -> Optional[Dict]:
        """Upload single image to Supabase storage + database"""
        if not self.is_connected():
            return None
        
        try:
            # DEBUG: Print file info
            #print(f"🔍 DEBUG - File name: {image_file.name}")
            #print(f"🔍 DEBUG - File type: {type(image_file)}")
            #print(f"🔍 DEBUG - File size: {len(image_file.getvalue())} bytes")
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            file_extension = image_file.name.split('.')[-1] if '.' in image_file.name else 'png'
            stored_filename = f"{source_page}_{timestamp}_{unique_id}.{file_extension}"
            storage_path = f"{source_page}/{stored_filename}"
            
            #print(f"🔍 DEBUG - Storage path: {storage_path}")
            
            # Reset file pointer (IMPORTANT!)
            image_file.seek(0)
            
            # Upload to Supabase storage bucket
            #print(f"🔍 DEBUG - Uploading to bucket: {self.bucket_name}")
            response = self.client.storage.from_(self.bucket_name).upload(
                storage_path, 
                image_file.getvalue(),
                file_options={"content-type": f"image/{file_extension}"}
            )
            
            # DEBUG: Print storage response
            #print(f"🔍 DEBUG - Storage response: {response}")
            #print(f"🔍 DEBUG - Response type: {type(response)}")
            
            # Check for storage errors
            if hasattr(response, 'error') and response.error:
                #print(f"🔍 DEBUG - Storage error: {response.error}")
                st.error(f"❌ Storage upload failed: {response.error}")
                return None
                
            # Check status code if available
            if hasattr(response, 'status_code'):
               # print(f"🔍 DEBUG - Status code: {response.status_code}")
                if response.status_code != 200:
                    st.error(f"❌ Storage upload failed with status: {response.status_code}")
                    return None
            
            # Get public URL
            public_url = self.client.storage.from_(self.bucket_name).get_public_url(storage_path)
           # print(f"🔍 DEBUG - Public URL: {public_url}")
            
            # Save metadata to database table
            upload_data = {
                'original_filename': image_file.name,
                'stored_filename': stored_filename,
                'storage_path': storage_path,
                'public_url': public_url,
                'file_size': len(image_file.getvalue()),
                'source_page': source_page,
                'upload_type': 'single',
                'session_id': self.get_session_id()
            }
            
            #print(f"🔍 DEBUG - About to insert: {upload_data}")
            
            # Reset file pointer again before database insert
            image_file.seek(0)
            
            db_response = self.client.table('image_uploads').insert(upload_data).execute()
            
            #print(f"🔍 DEBUG - DB response: {db_response}")
            
            if hasattr(db_response, 'error') and db_response.error:
              #  print(f"🔍 DEBUG - DB error: {db_response.error}")
                st.error(f"❌ Database error: {db_response.error}")
                return None
            
            if db_response.data:
                st.success(f"✅ Image stored: {image_file.name}")
                return {
                    'upload_id': db_response.data[0]['id'],
                    'public_url': public_url,
                    **upload_data
                }
            else:
                st.error("❌ Failed to save to database")
                return None
                
        except Exception as e:
          #  print(f"🔍 DEBUG - Exception: {str(e)}")
          #  print(f"🔍 DEBUG - Exception type: {type(e)}")
            st.error(f"❌ Error uploading: {str(e)}")
            return None
    
    def upload_batch(self, files_data: list, source_page: str) -> list:
        """Upload multiple images from ZIP folder"""
        if not self.is_connected():
            return []
        
        batch_id = str(uuid.uuid4())
        uploaded_files = []
        
        progress_bar = st.progress(0)
        st.info(f"📦 Storing {len(files_data)} images to Supabase...")
        
        for i, (file_content, original_name) in enumerate(files_data):
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                file_extension = original_name.split('.')[-1] if '.' in original_name else 'png'
                stored_filename = f"batch_{timestamp}_{unique_id}.{file_extension}"
                storage_path = f"{source_page}_batch/{stored_filename}"
                
                # Upload to storage
                response = self.client.storage.from_(self.bucket_name).upload(
                    storage_path,
                    file_content,
                    file_options={"content-type": f"image/{file_extension}"}
                )
                
                # Check for storage errors
                if hasattr(response, 'error') and response.error:
                   # print(f"🔍 BATCH DEBUG - Storage failed for {original_name}: {response.error}")
                    continue
                
                # Storage successful - get public URL
                public_url = self.client.storage.from_(self.bucket_name).get_public_url(storage_path)
                #print(f"🔍 BATCH DEBUG - Storage success for {original_name}, URL: {public_url}")
                
                # Save to database
                upload_data = {
                    'original_filename': original_name,
                    'stored_filename': stored_filename,
                    'storage_path': storage_path,
                    'public_url': public_url,
                    'file_size': len(file_content),
                    'source_page': source_page,
                    'upload_type': 'batch',
                    'batch_id': batch_id,
                    'session_id': self.get_session_id()
                }
                
                #print(f"🔍 BATCH DEBUG - About to insert to DB: {upload_data}")
                
                db_response = self.client.table('image_uploads').insert(upload_data).execute()
                
                #print(f"🔍 BATCH DEBUG - DB response: {db_response}")
                
                if hasattr(db_response, 'error') and db_response.error:
                   # print(f"🔍 BATCH DEBUG - DB error for {original_name}: {db_response.error}")
                    continue
                
                if db_response.data:
                   # print(f"🔍 BATCH DEBUG - DB success for {original_name}")
                    uploaded_files.append({
                        'upload_id': db_response.data[0]['id'],
                        'original_name': original_name,
                        **upload_data
                    })
                
                progress = (i + 1) / len(files_data)
                progress_bar.progress(progress)
                
            except Exception as e:
             #   print(f"🔍 BATCH DEBUG - Exception for {original_name}: {str(e)}")
                st.warning(f"⚠️ Failed: {original_name}: {str(e)}")
        
        progress_bar.empty()
        
        #print(f"🔍 BATCH DEBUG - Final results: {len(uploaded_files)} successful uploads")
        
        if uploaded_files:
            st.success(f"✅ Stored {len(uploaded_files)}/{len(files_data)} images to Supabase")
        else:
            st.warning(f"⚠️ No images were saved to database, but {len(files_data)} were uploaded to storage")
        
        return uploaded_files

# Global instance
supabase_storage = SupabaseImageStorage()