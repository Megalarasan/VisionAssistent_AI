import cv2
import numpy as np
import speech_recognition as sr
import pyttsx3
import threading
import time
import google.generativeai as genai
import PIL.Image
import re
from datetime import datetime


class VisionAIAssistant:
    def __init__(self, google_api_key):
        """
        Comprehensive Vision AI Assistant with Multi-Modal Intelligence
        """
        # Camera Setup
        self.camera = cv2.VideoCapture(0)
        self.camera.set(3, 2000)  # Set width
        self.camera.set(4, 1860)  # Set height

        # Speech Components Setup
        self.recognizer = None
        self.microphone = None
        self.speech_engine = None

        # AI Model Setup
        self.vision_model = None
        self.language_model = None

        # API Key
        self.google_api_key = google_api_key

        # Initialize Speech Components
        self.setup_speech_components()

        # AI Model Setup
        self.setup_ai_models(google_api_key)

        # Knowledge Management
        self.knowledge_base = {
            'current_context': {},
            'conversation_history': []
        }

        # Image Storage
        self.image_memory = []  # List to store multiple images with their details
        self.last_image_index = -1  # Index of the last referenced image

        # Processing Flags
        self.is_running = True
        self.is_listening = True

    def setup_speech_components(self):
        """
        Comprehensive setup of speech recognition and text-to-speech components
        """
        try:
            # Speech Recognition Setup
            self.recognizer = sr.Recognizer()
            self.recognizer.dynamic_energy_threshold = True

            # Microphone Setup
            try:
                self.microphone = sr.Microphone()
            except Exception as mic_error:
                print(f"Microphone setup error: {mic_error}")
                self.microphone = None

            # Text-to-Speech Engine Setup
            try:
                self.speech_engine = pyttsx3.init()
                self.configure_speech_engine()
            except Exception as tts_error:
                print(f"Text-to-speech setup error: {tts_error}")
                self.speech_engine = None

        except Exception as e:
            print(f"Comprehensive speech component setup failed: {e}")
            self.recognizer = None
            self.microphone = None
            self.speech_engine = None

    def configure_speech_engine(self):
        """
        Configure text-to-speech engine settings
        """
        if not self.speech_engine:
            return

        try:
            # Adjust speech rate
            self.speech_engine.setProperty('rate', 160)  # Speaking speed

            # Adjust volume
            self.speech_engine.setProperty('volume', 0.9)  # Volume level

            # Select voice (if multiple voices available)
            voices = self.speech_engine.getProperty('voices')
            if len(voices) > 1:
                # Try to select a more natural-sounding voice
                self.speech_engine.setProperty('voice', voices[1].id)
        except Exception as e:
            print(f"Speech engine configuration error: {e}")

    def speak(self, text):
        """
        Text-to-speech conversion with comprehensive error handling
        """
        try:
            # Validate speech engine
            if not self.speech_engine:
                print(f"Cannot speak (no TTS engine): {text}")
                return

            # Print and speak the text
            print(f"Speaking: {text}")
            self.speech_engine.say(text)
            self.speech_engine.runAndWait()

        except Exception as e:
            print(f"Speech output error: {e}")

    def setup_ai_models(self, google_api_key):
        """
        Configure AI models for vision and language understanding
        """
        try:
            # Configure Gemini API
            genai.configure(api_key=google_api_key)

            # Use Vision model
            self.vision_model = genai.GenerativeModel('gemini-2.0-flash')

            # Use Language model
            self.language_model = genai.GenerativeModel('gemini-2.0-flash')

        except Exception as e:
            print(f"Error setting up AI models: {e}")
            self.vision_model = None
            self.language_model = None

    def convert_frame_to_pil_image(self, frame):
        """
        Convert OpenCV frame to PIL Image
        """
        try:
            # Convert BGR (OpenCV) to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            pil_image = PIL.Image.fromarray(rgb_frame)

            return pil_image
        except Exception as e:
            print(f"Image conversion error: {e}")
            return None

    def capture_and_analyze_scene(self, command):
        """
        Capture current scene and analyze it comprehensively
        """
        try:
            # Capture frame
            ret, frame = self.camera.read()
            if not ret:
                self.speak("Unable to capture frame")
                return None

            # Display the captured frame
            cv2.imshow('Camera View', frame)
            cv2.waitKey(1)  # Wait for 1ms to display the frame

            # Convert frame to PIL Image
            pil_image = self.convert_frame_to_pil_image(frame)
            if not pil_image:
                self.speak("Image processing failed")
                return None

            # Generate scene description using AI
            prompt = command

            try:
                response = self.vision_model.generate_content([prompt, pil_image])
                scene_description = response.text if response else "No description available"

                # Clean the output text
                scene_description = self.clean_output_text(scene_description)

                # Create image metadata to store content information
                metadata_prompt = "Identify and list the main objects in this image. Format as a comma-separated list."

                try:
                    metadata_response = self.vision_model.generate_content([metadata_prompt, pil_image])
                    content_metadata = metadata_response.text if metadata_response else "Unknown content"
                except:
                    content_metadata = "Unknown content"

                # Store the image with its details
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                image_data = {
                    "image": pil_image,
                    "description": scene_description,
                    "content_metadata": content_metadata,
                    "timestamp": timestamp,
                    "original_command": command
                }

                self.image_memory.append(image_data)
                self.last_image_index = len(self.image_memory) - 1

                print(f"Stored image {self.last_image_index} with metadata: {content_metadata}")

            except Exception as ai_error:
                print(f"Scene analysis error: {ai_error}")
                scene_description = "Scene analysis failed due to an AI error."

            return scene_description

        except Exception as e:
            print(f"Comprehensive scene capture and analysis error: {e}")
            return None

    def clean_output_text(self, text):
        """
        Clean the output text by removing unwanted characters
        """
        text = text.replace("*", "")
        text = re.sub(r"\*\*", "", text)
        return text.strip()

    def analyze_last_image(self, command):
        """
        Analyze the last captured image with a new question
        """
        if len(self.image_memory) == 0 or self.last_image_index < 0:
            return "I don't have any previous image to analyze. Please capture a new image first."

        try:
            # Get the last referenced image
            image_data = self.image_memory[self.last_image_index]

            # Generate new analysis based on the stored image and new command
            prompt = f"Based on the previous image: {command}"

            try:
                response = self.vision_model.generate_content([prompt, image_data["image"]])
                scene_description = response.text if response else "No description available"

                # Clean the output text
                scene_description = self.clean_output_text(scene_description)

            except Exception as ai_error:
                print(f"Last image analysis error: {ai_error}")
                scene_description = "Analysis failed due to an AI error."

            return scene_description

        except Exception as e:
            print(f"Error analyzing last image: {e}")
            return "I encountered an error analyzing the previous image."

    def find_image_by_content(self, content_query):
        """
        Find an image in memory that contains specific content
        """
        if len(self.image_memory) == 0:
            return None, "I don't have any images stored. Please capture images first."

        try:
            # First try to find images by metadata
            matching_indices = []

            for i, image_data in enumerate(self.image_memory):
                # Check if the content query is in the metadata
                if content_query.lower() in image_data["content_metadata"].lower():
                    matching_indices.append(i)

            if not matching_indices:
                # If no matches found by metadata, reanalyze all images for the content
                for i, image_data in enumerate(self.image_memory):
                    prompt = f"Is there {content_query} in this image? Answer with just Yes or No."
                    try:
                        response = self.vision_model.generate_content([prompt, image_data["image"]])
                        answer = response.text.lower() if response else ""

                        if "yes" in answer:
                            matching_indices.append(i)
                    except Exception:
                        continue

            if matching_indices:
                # Return the most recent matching image
                best_index = matching_indices[-1]
                self.last_image_index = best_index
                return self.image_memory[best_index]["image"], f"Found image with {content_query}."
            else:
                return None, f"I couldn't find any image containing {content_query}."

        except Exception as e:
            print(f"Error finding image by content: {e}")
            return None, f"Error searching for images with {content_query}."

    def analyze_specific_image(self, content_query, analysis_command):
        """
        Find and analyze a specific image based on content description
        """
        image, message = self.find_image_by_content(content_query)

        if image is None:
            return message

        try:
            prompt = analysis_command

            response = self.vision_model.generate_content([prompt, image])
            scene_description = response.text if response else "No description available"

            # Clean the output text
            scene_description = self.clean_output_text(scene_description)

            return scene_description

        except Exception as e:
            print(f"Error analyzing specific image: {e}")
            return "I encountered an error analyzing the image."

    def process_command(self, command):
        """
        Process user commands with intelligent routing
        """
        try:
            # Convert command to lowercase for consistent matching
            command_lower = command.lower()

            # Check for shutdown commands
            shutdown_keywords = ['bye', 'finish', 'shut down', 'end', 'exit']
            if any(keyword in command_lower for keyword in shutdown_keywords):
                self.is_running = False
                self.is_listening = False
                self.speak("Shutting down vision assistant. Goodbye!")
                return False

            # Check for commands about the previous image
            previous_image_keywords = [
                "from the last image", "in the previous image", "from the previous image",
                "about the last image", "regarding the last image", "from that last picture",
                "in the last picture", "from the earlier image", "from the recent image",
                "in that image", "in the image you just saw"
            ]

            if any(keyword in command_lower for keyword in previous_image_keywords):
                if len(self.image_memory) == 0 or self.last_image_index < 0:
                    self.speak("I don't have any previous image to analyze. Please capture a new image first.")
                else:
                    # Get analysis of the last image with new question
                    scene_description = self.analyze_last_image(command)
                    if scene_description:
                        self.speak(scene_description)
                    else:
                        self.speak("I couldn't analyze the previous image with your question. Please try again.")
                return True

            # Check for command to analyze specific image by content
            specific_image_patterns = [
                r"take the images?\s+which has ([a-zA-Z0-9\s]+) and (.+)",
                r"take the images?\s+with ([a-zA-Z0-9\s]+) and (.+)",
                r"find the images?\s+with ([a-zA-Z0-9\s]+) and (.+)",
                r"find the images?\s+which has ([a-zA-Z0-9\s]+) and (.+)"
            ]

            for pattern in specific_image_patterns:
                match = re.search(pattern, command_lower)
                if match:
                    content_query = match.group(1).strip()
                    analysis_command = match.group(2).strip()
                    result = self.analyze_specific_image(content_query, analysis_command)
                    self.speak(result)
                    return True

            # Vision-related queries
            vision_keywords = [
                "who is he", "who is she", "identify this person", "from this picture", "in this picture",
                "name of this person",
                "tell me about this individual",
                "person details", "describe this person", "facial recognition", "who are they",
                "personal identification",
                "who is standing there", "who is in front of me", "who is behind me", "what is this",
                "what is this animal",
                "name of this animal", "animal species", "identify the animal", "animal type", "breed of this animal",
                "animal details", "describe this creature", "what kind of animal", "animal classification",
                "is there an animal near me", "what animal is in front of me", "where is this", "name of this place",
                "location details", "identify location", "what is this place", "describe this location", "place type",
                "geographic details", "location recognition", "environment description", "what city am I in",
                "where am I standing", "is this a safe place", "what is this object", "name this item", "object type",
                "identify object", "what kind of thing is this", "object details", "describe this object",
                "item classification",
                "what am I looking at", "object recognition", "what is near me", "what object is in front of me",
                "what is behind me", "what plant is this", "name of this plant", "plant species", "identify plant",
                "plant type",
                "botanical details", "describe this plant", "plant classification", "leaf identification",
                "flower recognition",
                "is this plant poisonous", "can I eat this plant", "what food is this", "name this dish", "food type",
                "identify cuisine", "describe this meal", "food details", "ingredient recognition",
                "culinary identification",
                "meal type", "food classification", "is this food fresh", "does this food look safe to eat",
                "what vehicle is this",
                "car model", "vehicle type", "identify vehicle", "car details", "vehicle brand", "transportation mode",
                "vehicle recognition", "describe this vehicle", "vehicle classification", "is this a car or a bike",
                "what kind of vehicle is in front of me", "is a vehicle coming toward me", "what landmark is this",
                "identify landmark", "famous location", "landmark details", "name this place", "historical site",
                "architectural recognition", "location landmark", "describe this landmark",
                "tourist spot identification",
                "is this place famous", "have I been here before", "what clothing is this", "identify outfit",
                "clothing type",
                "fashion details", "dress description", "clothing style", "outfit recognition", "garment type",
                "describe this clothing", "fashion classification", "is this formal wear",
                "what color is this clothing",
                "what is this", "identify this", "tell me about this", "details of this", "describe what I'm seeing",
                "provide information", "explain this", "give me context", "what am I looking at",
                "comprehensive description",
                "what is around me", "analyze my surroundings", "what did you see", "describe the environment",
                "capture the image", "analyze the scene", "what is beside me", "what is that"
            ]

            if any(keyword in command_lower for keyword in vision_keywords):
                scene_description = self.capture_and_analyze_scene(command)
                if scene_description:
                    self.speak(scene_description)
                else:
                    self.speak("I couldn't analyze the scene. Please try again.")
                return True

            # Check if user wants to list stored images
            if any(keyword in command_lower for keyword in
                   ["list images", "list all images", "show stored images", "what images do you have"]):
                if len(self.image_memory) == 0:
                    self.speak("I don't have any images stored yet.")
                else:
                    image_list = f"I have {len(self.image_memory)} images stored:"
                    for i, img_data in enumerate(self.image_memory):
                        image_list += f"\nImage {i + 1}: {img_data['content_metadata']} (captured at {img_data['timestamp']})"
                    self.speak(image_list)
                return True

            # General knowledge queries
            try:
                response = self.language_model.generate_content(command)
                ai_response = response.text if response else "I couldn't generate a response."

                # Clean the output text
                ai_response = self.clean_output_text(ai_response)

                self.speak(ai_response)
            except Exception as e:
                print(f"General query processing error: {e}")
                self.speak("I apologize, but I couldn't process your query.")
            return True

        except Exception as e:
            print(f"Command processing error: {e}")
            self.speak("An error occurred while processing your command.")
            return True

    def start_voice_interface(self):
        """
        Continuous voice command processing with robust error handling
        """
        # Check if speech recognition is available
        if not self.recognizer or not self.microphone:
            print("Speech recognition not available. Skipping voice interface.")
            return

        self.is_running = True
        self.is_listening = True
        self.speak("Vision AI Assistant is initialized and ready. How can I help you?")

        while self.is_listening:
            try:
                # Use microphone as audio source
                with self.microphone as source:
                    print("Listening for commands...")

                    # Adjust for ambient noise
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

                    # Set energy threshold dynamically
                    self.recognizer.dynamic_energy_threshold = True

                    # Listen for audio input
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)

                    # Attempt to recognize speech
                    try:
                        command = self.recognizer.recognize_google(audio).lower()
                        print(f"Recognized command: {command}")

                        # Process the recognized command
                        continue_listening = self.process_command(command)
                        if not continue_listening:
                            break

                    except sr.UnknownValueError:
                        self.speak("Sorry, I didn't catch that. Could you repeat?")
                    except sr.RequestError as e:
                        print(f"Could not request results: {e}")
                        self.speak("There was an issue with speech recognition.")
                    except Exception as e:
                        print(f"Unexpected speech recognition error: {e}")
                        self.speak("An unexpected error occurred.")

            except Exception as e:
                print(f"Voice interface error: {e}")
                time.sleep(2)  # Prevent tight error loops

        # Release OpenCV windows
        cv2.destroyAllWindows()

    def start_vision_assistant(self):
        """
        Start the Vision AI Assistant
        """
        try:
            # Start voice interface
            voice_thread = threading.Thread(target=self.start_voice_interface)
            voice_thread.start()

            # Keep the main thread running
            voice_thread.join()

        except Exception as e:
            print(f"Vision assistant startup error: {e}")
        finally:
            # Cleanup
            self.camera.release()

def main():
    # Replace with your actual Google API key
    GOOGLE_API_KEY = 'AIzaSyAL0EJGSp-g7rhuBwQpgk8T95llLa6kq1c'

    # Initialize Vision AI Assistant
    vision_assistant = VisionAIAssistant(GOOGLE_API_KEY)

    try:
        # Start the assistant directly in listening mode
        vision_assistant.start_voice_interface()

    except KeyboardInterrupt:
        print("Vision AI Assistant Stopped by User")
    finally:
        vision_assistant.is_running = False
        vision_assistant.is_listening = False
        vision_assistant.camera.release()

if __name__ == "__main__":
    main()