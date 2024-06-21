# import whisper
# import os
# import shutil
# import cv2
# from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip
# from tqdm import tqdm

# FONT = cv2.FONT_HERSHEY_SIMPLEX
# FONT_SCALE = 0.8
# FONT_THICKNESS = 2

# class VideoTranscriber:
#     def __init__(self, model_path, video_path):
#         self.model = whisper.load_model(model_path)
#         self.video_path = video_path
#         self.audio_path = ''
#         self.text_array = []
#         self.fps = 0
#         self.char_width = 0

#     def transcribe_video(self):
#         print('Transcribing video')
#         result = self.model.transcribe(self.audio_path)
#         text = result["segments"][0]["text"]
#         textsize = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
#         cap = cv2.VideoCapture(self.video_path)
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         asp = 16/9
#         ret, frame = cap.read()
#         width = frame[:, int(int(width - 1 / asp * height) / 2):width - int((width - 1 / asp * height) / 2)].shape[1]
#         width = width - (width * 0.1)
#         self.fps = cap.get(cv2.CAP_PROP_FPS)
#         self.char_width = int(textsize[0] / len(text))
        
#         for j in tqdm(result["segments"]):
#             lines = []
#             text = j["text"]
#             end = j["end"]
#             start = j["start"]
#             total_frames = int((end - start) * self.fps)
#             start = start * self.fps
#             total_chars = len(text)
#             words = text.split(" ")
#             i = 0
            
#             while i < len(words):
#                 words[i] = words[i].strip()
#                 if words[i] == "":
#                     i += 1
#                     continue
#                 length_in_pixels = (len(words[i]) + 1) * self.char_width
#                 remaining_pixels = width - length_in_pixels
#                 line = words[i] 
                
#                 while remaining_pixels > 0:
#                     i += 1 
#                     if i >= len(words):
#                         break
#                     length_in_pixels = (len(words[i]) + 1) * self.char_width
#                     remaining_pixels -= length_in_pixels
#                     if remaining_pixels < 0:
#                         continue
#                     else:
#                         line += " " + words[i]
                
#                 line_array = [line, int(start) + 15, int(len(line) / total_chars * total_frames) + int(start) + 15]
#                 start = int(len(line) / total_chars * total_frames) + int(start)
#                 lines.append(line_array)
#                 self.text_array.append(line_array)
        
#         cap.release()
#         print('Transcription complete')
    
#     def extract_audio(self):
#         print('Extracting audio')
#         audio_path = os.path.join(os.path.dirname(self.video_path), "audio.mp3")
#         video = VideoFileClip(self.video_path)
#         audio = video.audio 
#         audio.write_audiofile(audio_path)
#         self.audio_path = audio_path
#         print('Audio extracted')
    
#     def extract_frames(self, output_folder):
#         print('Extracting frames')
#         cap = cv2.VideoCapture(self.video_path)
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         asp = width / height
#         N_frames = 0
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame = frame[:, int(int(width - 1 / asp * height) / 2):width - int((width - 1 / asp * height) / 2)]
            
#             for i in self.text_array:
#                 if N_frames >= i[1] and N_frames <= i[2]:
#                     text = i[0]
#                     text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
#                     text_x = int((frame.shape[1] - text_size[0]) / 2)
#                     text_y = int(height/2)
#                     cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
#                     break
            
#             cv2.imwrite(os.path.join(output_folder, str(N_frames) + ".jpg"), frame)
#             N_frames += 1
        
#         cap.release()
#         print('Frames extracted')

#     def create_video(self, output_video_path):
#         print('Creating video')
#         image_folder = os.path.join(os.path.dirname(self.video_path), "frames")
#         if not os.path.exists(image_folder):
#             os.makedirs(image_folder)
        
#         self.extract_frames(image_folder)
        
#         images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
#         images.sort(key=lambda x: int(x.split(".")[0]))
        
#         frame = cv2.imread(os.path.join(image_folder, images[0]))
#         height, width, layers = frame.shape
        
#         clip = ImageSequenceClip([os.path.join(image_folder, image) for image in images], fps=self.fps)
#         audio = AudioFileClip(self.audio_path)
#         clip = clip.set_audio(audio)
#         clip.write_videofile(output_video_path)
#         shutil.rmtree(image_folder)
#         os.remove(os.path.join(os.path.dirname(self.video_path), "audio.mp3"))


# class GifCreator:
#     def __init__(self, video_path):
#         self.video_path = video_path

#     def create_gif(self, start_time, end_time, output_gif_path):
#         try:
#             print(f'Creating GIF from {start_time}s to {end_time}s')
#             video = VideoFileClip(self.video_path).subclip(start_time, end_time)
#             gif_clip = video.to_gif()
#             gif_clip.write_gif(output_gif_path)
#             print(f'GIF created at {output_gif_path}')
#         except Exception as e:
#             print(f"Error creating GIF: {e}")

# # Example usage
# model_path = "base"
# video_path = "test/creategif.mp4"
# output_video_path = "output.mp4"
# output_audio_path = "test/audio.mp3"
    
# transcriber = VideoTranscriber(model_path, video_path)

# # Extract audio from the video
# transcriber.extract_audio()

# # Transcribe the audio and prepare text for overlay
# transcriber.transcribe_video()

# # Create a new video with the transcribed text overlay
# transcriber.create_video(output_video_path)


import whisper
import os
import shutil
import cv2
from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import scenedetect
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2

class VideoTranscriber:
    def __init__(self, model_path, video_path):
        self.model = whisper.load_model(model_path)
        self.video_path = video_path
        self.audio_path = ''
        self.text_array = []
        self.fps = 0
        self.char_width = 0

    def transcribe_video(self):
        print('Transcribing video')
        result = self.model.transcribe(self.audio_path)
        text = result["segments"][0]["text"]
        textsize = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        asp = 16 / 9
        ret, frame = cap.read()
        width = frame[:, int(int(width - 1 / asp * height) / 2):width - int((width - 1 / asp * height) / 2)].shape[1]
        width = width - (width * 0.1)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.char_width = int(textsize[0] / len(text))
        
        for j in tqdm(result["segments"]):
            lines = []
            text = j["text"]
            end = j["end"]
            start = j["start"]
            total_frames = int((end - start) * self.fps)
            start = start * self.fps
            total_chars = len(text)
            words = text.split(" ")
            i = 0
            
            while i < len(words):
                words[i] = words[i].strip()
                if words[i] == "":
                    i += 1
                    continue
                length_in_pixels = (len(words[i]) + 1) * self.char_width
                remaining_pixels = width - length_in_pixels
                line = words[i] 
                
                while remaining_pixels > 0:
                    i += 1 
                    if i >= len(words):
                        break
                    length_in_pixels = (len(words[i]) + 1) * self.char_width
                    remaining_pixels -= length_in_pixels
                    if remaining_pixels < 0:
                        continue
                    else:
                        line += " " + words[i]
                
                line_array = [line, int(start) + 15, int(len(line) / total_chars * total_frames) + int(start) + 15]
                start = int(len(line) / total_chars * total_frames) + int(start)
                lines.append(line_array)
                self.text_array.append(line_array)
        
        cap.release()
        print('Transcription complete')
    
    def extract_audio(self):
        print('Extracting audio')
        audio_path = os.path.join(os.path.dirname(self.video_path), "audio.mp3")
        try:
            video = VideoFileClip(self.video_path)
            audio = video.audio
            audio.write_audiofile(audio_path)
            self.audio_path = audio_path
            print('Audio extracted')
        except Exception as e:
            print(f"Error extracting audio: {e}")
    
    def extract_frames(self, output_folder):
        print('Extracting frames')
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        asp = width / height
        N_frames = 0
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = frame[:, int(int(width - 1 / asp * height) / 2):width - int((width - 1 / asp * height) / 2)]
            
            for i in self.text_array:
                if N_frames >= i[1] and N_frames <= i[2]:
                    text = i[0]
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    text_x = int((frame.shape[1] - text_size[0]) / 2)
                    text_y = int(height / 2)
                    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    break
            
            cv2.imwrite(os.path.join(output_folder, str(N_frames) + ".jpg"), frame)
            N_frames += 1
        
        cap.release()
        print('Frames extracted')

    def create_video(self, output_video_path):
        print('Creating video')
        image_folder = os.path.join(os.path.dirname(self.video_path), "frames")
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        
        self.extract_frames(image_folder)
        
        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        images.sort(key=lambda x: int(x.split(".")[0]))
        
        try:
            clip = ImageSequenceClip([os.path.join(image_folder, image) for image in images], fps=self.fps)
            audio = AudioFileClip(self.audio_path)
            clip = clip.set_audio(audio)
            clip.write_videofile(output_video_path)
        except Exception as e:
            print(f"Error creating video: {e}")
        finally:
            shutil.rmtree(image_folder)
            if os.path.exists(self.audio_path):
                os.remove(self.audio_path)

class GifCreator:
    def __init__(self, video_path):
        self.video_path = video_path

    def detect_scenes(self):
        print('Detecting scenes')
        video_manager = VideoManager([self.video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())
        
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()

        video_manager.release()
        print(f'{len(scene_list)} scenes detected')
        return scene_list

    def create_gif(self, start_time, end_time, output_gif_path):
        try:
            print(f'Creating GIF from {start_time}s to {end_time}s')
            video = VideoFileClip(self.video_path).subclip(start_time, end_time)
            video.write_gif(output_gif_path, fps=10)
            print(f'GIF created at {output_gif_path}')
        except Exception as e:
            print(f"Error creating GIF: {e}")

    def create_gifs_from_scenes(self, output_folder):
        scenes = self.detect_scenes()
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        for i, scene in enumerate(scenes):
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            output_gif_path = os.path.join(output_folder, f"scene_{i + 1}.gif")
            self.create_gif(start_time, end_time, output_gif_path)

# Example usage for VideoTranscriber
model_path = "base"
video_path = "test/creategif.mp4"
output_video_path = "output.mp4"

# Create an instance of VideoTranscriber
transcriber = VideoTranscriber(model_path, video_path)

# Extract audio from the video
transcriber.extract_audio()

# Transcribe the audio and prepare text for overlay
transcriber.transcribe_video()

# Create a new video with the transcribed text overlay
transcriber.create_video(output_video_path)

# Example usage for GifCreator
gif_creator = GifCreator(output_video_path)

# Create GIFs for each detected scene
output_gif_folder = "gifs"
gif_creator.create_gifs_from_scenes(output_gif_folder)
