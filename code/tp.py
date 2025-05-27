import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import numpy as np
from PIL import Image, ImageTk
import io

# Try to import optional modules
try:
    import librosa
    import soundfile as sf
    import noisereduce as nr
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class BackgroundRemoverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Background & Music Remover")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c3e50')
        
        # Variables
        self.current_image = None
        self.current_audio = None
        self.sample_rate = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main title
        title = tk.Label(self.root, text="AI Background & Music Remover", 
                        font=('Arial', 24, 'bold'), fg='white', bg='#2c3e50')
        title.pack(pady=20)
        
        # Status frame
        status_frame = tk.Frame(self.root, bg='#2c3e50')
        status_frame.pack(fill='x', padx=20, pady=5)
        
        # Module status
        status_text = "Available modules: "
        modules = []
        if REMBG_AVAILABLE:
            modules.append("✅ AI Background Removal")
        else:
            modules.append("❌ AI Background Removal (install: pip install rembg)")
            
        if AUDIO_AVAILABLE:
            modules.append("✅ Audio Processing")
        else:
            modules.append("❌ Audio Processing (install: pip install librosa soundfile noisereduce)")
            
        if CV2_AVAILABLE:
            modules.append("✅ Basic Image Processing")
        else:
            modules.append("❌ OpenCV (install: pip install opencv-python)")
        
        status_label = tk.Label(status_frame, text="\n".join(modules), 
                               font=('Arial', 10), fg='yellow', bg='#2c3e50', justify='left')
        status_label.pack(anchor='w')
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Image processing tab
        self.image_frame = tk.Frame(notebook, bg='#34495e')
        notebook.add(self.image_frame, text="Image Background Removal")
        self.setup_image_tab()
        
        # Audio processing tab
        if AUDIO_AVAILABLE:
            self.audio_frame = tk.Frame(notebook, bg='#34495e')
            notebook.add(self.audio_frame, text="Audio Noise Removal")
            self.setup_audio_tab()
        else:
            disabled_frame = tk.Frame(notebook, bg='#34495e')
            notebook.add(disabled_frame, text="Audio (Unavailable)")
            error_label = tk.Label(disabled_frame, 
                                 text="Audio processing unavailable.\nInstall required packages:\npip install librosa soundfile noisereduce", 
                                 font=('Arial', 14), fg='red', bg='#34495e', justify='center')
            error_label.pack(expand=True)
    
    def setup_image_tab(self):
        # Image controls frame
        controls_frame = tk.Frame(self.image_frame, bg='#34495e')
        controls_frame.pack(fill='x', padx=20, pady=10)
        
        # Upload button
        upload_btn = tk.Button(controls_frame, text="Upload Image", 
                              command=self.upload_image, font=('Arial', 12),
                              bg='#3498db', fg='white', padx=20, pady=5)
        upload_btn.pack(side='left', padx=5)
        
        # Remove background button
        if REMBG_AVAILABLE:
            remove_btn = tk.Button(controls_frame, text="AI Remove Background", 
                                  command=self.remove_image_background_ai, font=('Arial', 12),
                                  bg='#e74c3c', fg='white', padx=20, pady=5)
            remove_btn.pack(side='left', padx=5)
        
        # Basic background removal (without AI)
        if CV2_AVAILABLE:
            basic_btn = tk.Button(controls_frame, text="Basic Remove Background", 
                                 command=self.remove_background_basic, font=('Arial', 12),
                                 bg='#9b59b6', fg='white', padx=20, pady=5)
            basic_btn.pack(side='left', padx=5)
        
        # Replace background button
        replace_btn = tk.Button(controls_frame, text="Replace Background", 
                               command=self.replace_background, font=('Arial', 12),
                               bg='#f39c12', fg='white', padx=20, pady=5)
        replace_btn.pack(side='left', padx=5)
        
        # Save button
        save_btn = tk.Button(controls_frame, text="Save Image", 
                            command=self.save_image, font=('Arial', 12),
                            bg='#27ae60', fg='white', padx=20, pady=5)
        save_btn.pack(side='left', padx=5)
        
        # Image display frame
        self.image_display_frame = tk.Frame(self.image_frame, bg='#34495e')
        self.image_display_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create canvas for images
        canvas_frame = tk.Frame(self.image_display_frame, bg='#34495e')
        canvas_frame.pack(fill='both', expand=True)
        
        # Original image side
        left_frame = tk.Frame(canvas_frame, bg='#34495e')
        left_frame.pack(side='left', fill='both', expand=True, padx=10)
        
        tk.Label(left_frame, text="Original Image", font=('Arial', 12), 
                fg='white', bg='#34495e').pack(pady=5)
        
        self.original_canvas = tk.Canvas(left_frame, bg='#2c3e50', width=400, height=300)
        self.original_canvas.pack(pady=5)
        
        # Processed image side
        right_frame = tk.Frame(canvas_frame, bg='#34495e')
        right_frame.pack(side='right', fill='both', expand=True, padx=10)
        
        tk.Label(right_frame, text="Processed Image", font=('Arial', 12), 
                fg='white', bg='#34495e').pack(pady=5)
        
        self.processed_canvas = tk.Canvas(right_frame, bg='#2c3e50', width=400, height=300)
        self.processed_canvas.pack(pady=5)
    
    def setup_audio_tab(self):
        # Audio controls frame
        audio_controls = tk.Frame(self.audio_frame, bg='#34495e')
        audio_controls.pack(fill='x', padx=20, pady=10)
        
        # Upload audio button
        upload_audio_btn = tk.Button(audio_controls, text="Upload Audio", 
                                    command=self.upload_audio, font=('Arial', 12),
                                    bg='#3498db', fg='white', padx=20, pady=5)
        upload_audio_btn.pack(side='left', padx=5)
        
        # Remove noise button
        denoise_btn = tk.Button(audio_controls, text="Remove Noise", 
                               command=self.remove_audio_noise, font=('Arial', 12),
                               bg='#e74c3c', fg='white', padx=20, pady=5)
        denoise_btn.pack(side='left', padx=5)
        
        # Save audio button
        save_audio_btn = tk.Button(audio_controls, text="Save Audio", 
                                  command=self.save_audio, font=('Arial', 12),
                                  bg='#27ae60', fg='white', padx=20, pady=5)
        save_audio_btn.pack(side='left', padx=5)
        
        # Audio info display
        self.audio_info = tk.Text(self.audio_frame, height=15, width=80, 
                                 font=('Arial', 10), bg='#2c3e50', fg='white')
        self.audio_info.pack(padx=20, pady=20)
        
        # Progress bar
        self.audio_progress = ttk.Progressbar(self.audio_frame, mode='indeterminate')
        self.audio_progress.pack(fill='x', padx=20, pady=10)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif")]
        )
        if file_path:
            try:
                self.current_image = Image.open(file_path)
                self.display_image_on_canvas(self.current_image, self.original_canvas)
                messagebox.showinfo("Success", "Image uploaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def remove_image_background_ai(self):
        if not REMBG_AVAILABLE:
            messagebox.showerror("Error", "rembg not installed! Run: pip install rembg")
            return
            
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return
        
        def process():
            try:
                # Convert PIL image to bytes
                img_byte_arr = io.BytesIO()
                self.current_image.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                
                # Remove background using rembg
                output = remove(img_bytes)
                
                # Convert back to PIL image
                self.processed_image = Image.open(io.BytesIO(output))
                
                # Display processed image
                self.root.after(0, lambda: self.display_image_on_canvas(self.processed_image, self.processed_canvas))
                self.root.after(0, lambda: messagebox.showinfo("Success", "AI background removed successfully!"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to remove background: {str(e)}"))
        
        threading.Thread(target=process, daemon=True).start()
    
    def remove_background_basic(self):
        if not CV2_AVAILABLE:
            messagebox.showerror("Error", "OpenCV not installed! Run: pip install opencv-python")
            return
            
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return
        
        def process():
            try:
                # Convert PIL to OpenCV format
                cv_image = cv2.cvtColor(np.array(self.current_image), cv2.COLOR_RGB2BGR)
                
                # Simple background removal using GrabCut
                height, width = cv_image.shape[:2]
                
                # Create mask
                mask = np.zeros((height, width), np.uint8)
                
                # Define rectangle around the subject (simple heuristic)
                rect = (50, 50, width-100, height-100)
                
                # Initialize foreground and background models
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                
                # Apply GrabCut
                cv2.grabCut(cv_image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
                
                # Create final mask
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                
                # Apply mask
                result = cv_image * mask2[:, :, np.newaxis]
                
                # Convert back to PIL with transparency
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                self.processed_image = Image.fromarray(result_rgb)
                
                # Add alpha channel for transparency
                self.processed_image = self.processed_image.convert('RGBA')
                datas = self.processed_image.getdata()
                new_data = []
                for item in datas:
                    if item[0] == 0 and item[1] == 0 and item[2] == 0:
                        new_data.append((0, 0, 0, 0))  # Transparent
                    else:
                        new_data.append(item)
                self.processed_image.putdata(new_data)
                
                self.root.after(0, lambda: self.display_image_on_canvas(self.processed_image, self.processed_canvas))
                self.root.after(0, lambda: messagebox.showinfo("Success", "Basic background removal completed!"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to remove background: {str(e)}"))
        
        threading.Thread(target=process, daemon=True).start()
    
    def replace_background(self):
        if not hasattr(self, 'processed_image'):
            messagebox.showwarning("Warning", "Please remove background first!")
            return
        
        bg_path = filedialog.askopenfilename(
            title="Select Background Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if bg_path:
            try:
                background = Image.open(bg_path)
                background = background.resize(self.processed_image.size)
                
                # Create composite image
                if self.processed_image.mode != 'RGBA':
                    self.processed_image = self.processed_image.convert('RGBA')
                
                composite = Image.new('RGBA', self.processed_image.size, (255, 255, 255, 255))
                composite.paste(background, (0, 0))
                composite.paste(self.processed_image, (0, 0), self.processed_image)
                
                self.processed_image = composite
                self.display_image_on_canvas(self.processed_image, self.processed_canvas)
                messagebox.showinfo("Success", "Background replaced successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to replace background: {str(e)}")
    
    def display_image_on_canvas(self, image, canvas):
        # Resize image to fit canvas
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 400, 300
        
        # Calculate display size maintaining aspect ratio
        img_width, img_height = image.size
        ratio = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        display_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(display_image)
        
        # Clear canvas and display image
        canvas.delete("all")
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        canvas.create_image(x, y, anchor='nw', image=self.photo)
    
    def save_image(self):
        if not hasattr(self, 'processed_image'):
            messagebox.showwarning("Warning", "No processed image to save!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if file_path:
            try:
                # Convert RGBA to RGB if saving as JPEG
                if file_path.lower().endswith('.jpg') or file_path.lower().endswith('.jpeg'):
                    if self.processed_image.mode == 'RGBA':
                        rgb_image = Image.new('RGB', self.processed_image.size, (255, 255, 255))
                        rgb_image.paste(self.processed_image, mask=self.processed_image.split()[-1])
                        rgb_image.save(file_path)
                    else:
                        self.processed_image.save(file_path)
                else:
                    self.processed_image.save(file_path)
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def upload_audio(self):
        if not AUDIO_AVAILABLE:
            messagebox.showerror("Error", "Audio libraries not installed!")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg")]
        )
        if file_path:
            try:
                self.current_audio, self.sample_rate = librosa.load(file_path, sr=None)
                duration = len(self.current_audio) / self.sample_rate
                
                info_text = f"Audio loaded successfully!\n"
                info_text += f"File: {os.path.basename(file_path)}\n"
                info_text += f"Duration: {duration:.2f} seconds\n"
                info_text += f"Sample Rate: {self.sample_rate} Hz\n"
                
                self.audio_info.delete(1.0, tk.END)
                self.audio_info.insert(tk.END, info_text)
                messagebox.showinfo("Success", "Audio uploaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load audio: {str(e)}")
    
    def remove_audio_noise(self):
        if not AUDIO_AVAILABLE:
            messagebox.showerror("Error", "Audio libraries not installed!")
            return
            
        if self.current_audio is None:
            messagebox.showwarning("Warning", "Please upload an audio file first!")
            return
        
        def process():
            try:
                self.audio_progress.start()
                
                # Apply noise reduction
                self.processed_audio = nr.reduce_noise(y=self.current_audio, sr=self.sample_rate)
                
                self.audio_progress.stop()
                
                info_text = self.audio_info.get(1.0, tk.END)
                info_text += "\nNoise reduction applied successfully!"
                
                self.root.after(0, lambda: self.audio_info.delete(1.0, tk.END))
                self.root.after(0, lambda: self.audio_info.insert(tk.END, info_text))
                self.root.after(0, lambda: messagebox.showinfo("Success", "Noise removed successfully!"))
            except Exception as e:
                self.audio_progress.stop()
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to remove noise: {str(e)}"))
        
        threading.Thread(target=process, daemon=True).start()
    
    def save_audio(self):
        if not AUDIO_AVAILABLE:
            messagebox.showerror("Error", "Audio libraries not installed!")
            return
            
        if not hasattr(self, 'processed_audio'):
            messagebox.showwarning("Warning", "No processed audio to save!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Audio",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("FLAC files", "*.flac")]
        )
        if file_path:
            try:
                sf.write(file_path, self.processed_audio, self.sample_rate)
                messagebox.showinfo("Success", "Audio saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save audio: {str(e)}")

def main():
    root = tk.Tk()
    app = BackgroundRemoverApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()