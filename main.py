import sounddevice as sd
import numpy as np
import pyautogui
import time
import threading
from collections import deque

class GuitarMouseClicker:
    def __init__(self):
        # Audio settings
        self.CHUNK = 1024  # Number of audio samples per frame
        self.CHANNELS = 1  # Mono
        self.RATE = 44100  # Sample rate
        self.DTYPE = np.float32
        
        # Detection settings (adjust these to fine-tune sensitivity)
        self.THRESHOLD = 0.01  # Volume threshold to trigger click (0.0-1.0)
        self.DEBOUNCE_TIME = 0.2  # Minimum time between clicks (seconds)
        self.WINDOW_SIZE = 5  # Number of recent volume readings to average
        
        # State variables
        self.is_running = False
        self.last_click_time = 0
        self.volume_history = deque(maxlen=self.WINDOW_SIZE)
        self.audio_stream = None
        
        # Threading
        self.stop_event = threading.Event()
        
        # Initialize audio
        self.setup_audio()
    
    def setup_audio(self):
        """Initialize sounddevice and find Rocksmith cable"""
        print("Available audio devices:")
        devices = sd.query_devices()
        
        device_index = None
        
        for i, device in enumerate(devices):
            # Only show input devices
            if device['max_input_channels'] > 0:
                print(f"{i}: {device['name']} - Inputs: {device['max_input_channels']} - Sample Rate: {device['default_samplerate']}")
                
                # Look for Rocksmith cable (common names)
                device_name_lower = device['name'].lower()
                if any(keyword in device_name_lower for keyword in 
                       ['rocksmith', 'usb', 'guitar', 'line', 'real', 'tone', 'cable']):
                    device_index = i
                    print(f"*** Found potential Rocksmith device: {device['name']} ***")
        
        if device_index is None:
            print("\nCouldn't auto-detect Rocksmith cable.")
            print("Please look at the list above and find your Rocksmith cable.")
            try:
                device_index = int(input("Enter the device number for your Rocksmith cable: "))
            except ValueError:
                print("Invalid input. Using default input device.")
                device_index = sd.default.device[0]  # Default input device
        
        self.device_index = device_index
        self.device_info = devices[device_index]
        print(f"\nUsing device: {self.device_info['name']}")
        
        # Use the device's preferred sample rate
        preferred_rate = int(self.device_info['default_samplerate'])
        if preferred_rate != self.RATE:
            print(f"Adjusting sample rate from {self.RATE} to {preferred_rate}")
            self.RATE = preferred_rate
    
    def calculate_volume(self, audio_data):
        """Calculate RMS volume from audio data"""
        if len(audio_data) == 0:
            return 0
        return np.sqrt(np.mean(audio_data ** 2))
    
    def should_click(self, current_volume):
        """Determine if we should trigger a mouse click"""
        current_time = time.time()
        
        # Add current volume to history
        self.volume_history.append(current_volume)
        
        # Check if current volume is above threshold
        # and enough time has passed since last click
        volume_spike = current_volume > self.THRESHOLD
        time_passed = (current_time - self.last_click_time) > self.DEBOUNCE_TIME
        
        if volume_spike and time_passed:
            self.last_click_time = current_time
            return True
        
        return False
    
    def audio_callback(self, indata, frames, time, status):
        """Audio callback function - called automatically by sounddevice"""
        if status:
            print(f"Audio status: {status}")
        
        if self.stop_event.is_set():
            return
        
        # Convert to 1D array if stereo
        if indata.ndim > 1:
            audio_data = indata[:, 0]  # Take left channel
        else:
            audio_data = indata.flatten()
        
        # Calculate volume
        volume = self.calculate_volume(audio_data)
        
        # Check if we should click
        if self.should_click(volume):
            # Schedule click in main thread (GUI operations should be on main thread)
            threading.Thread(target=self.click_mouse, args=(volume,), daemon=True).start()
    
    def click_mouse(self, volume):
        """Perform mouse click - run in separate thread"""
        try:
            pyautogui.click()
            print(f"CLICK! (Volume: {volume:.4f})")
        except Exception as e:
            print(f"Click error: {e}")
    
    def start(self):
        """Start the guitar-to-click detection"""
        if self.is_running:
            print("Already running!")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        print(f"Listening for guitar input... (Threshold: {self.THRESHOLD})")
        print("Strum your guitar to trigger mouse clicks!")
        print("Press Ctrl+C to stop")
        
        try:
            # Start audio stream
            with sd.InputStream(
                device=self.device_index,
                channels=self.CHANNELS,
                samplerate=self.RATE,
                blocksize=self.CHUNK,
                dtype=self.DTYPE,
                callback=self.audio_callback
            ):
                # Keep running until interrupted
                while self.is_running and not self.stop_event.is_set():
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"Audio stream error: {e}")
            print("\nTroubleshooting tips:")
            print("1. Make sure your Rocksmith cable is plugged in")
            print("2. Try selecting a different device number")
            print("3. Check Windows Sound settings - make sure the cable shows input levels")
            print("4. Try running as administrator")
        
        except KeyboardInterrupt:
            pass
        
        finally:
            self.stop()
    
    def stop(self):
        """Stop the detection"""
        print("\nStopping...")
        self.is_running = False
        self.stop_event.set()
        print("Stopped!")
    
    def adjust_sensitivity(self, new_threshold):
        """Adjust the volume threshold for click detection"""
        self.THRESHOLD = max(0.001, min(1.0, new_threshold))
        print(f"Sensitivity adjusted to: {self.THRESHOLD}")
    
    def test_audio_levels(self, duration=10):
        """Test audio input levels without clicking"""
        print(f"Testing audio levels for {duration} seconds...")
        print("Play your guitar to see volume levels:")
        
        def test_callback(indata, frames, time, status):
            if indata.ndim > 1:
                audio_data = indata[:, 0]
            else:
                audio_data = indata.flatten()
            
            volume = self.calculate_volume(audio_data)
            # Show volume with visual indicator
            bar_length = int(volume * 50)  # Scale for display
            bar = "█" * bar_length + "░" * (50 - bar_length)
            threshold_pos = int(self.THRESHOLD * 50)
            
            print(f"\rVolume: {volume:.4f} |{bar}| Threshold at pos {threshold_pos}", end="")
        
        try:
            with sd.InputStream(
                device=self.device_index,
                channels=self.CHANNELS,
                samplerate=self.RATE,
                blocksize=self.CHUNK,
                dtype=self.DTYPE,
                callback=test_callback
            ):
                time.sleep(duration)
            
            print(f"\nTest complete! Use threshold around the volume you saw while strumming.")
            
        except Exception as e:
            print(f"Test failed: {e}")


def main():
    print("=== Guitar to Mouse Clicker (SoundDevice Version) ===")
    print("This script converts your guitar strums into mouse clicks!")
    print()
    
    try:
        clicker = GuitarMouseClicker()
    except Exception as e:
        print(f"Setup failed: {e}")
        print("\nMake sure you have installed: pip install sounddevice numpy pyautogui")
        return
    
    # Menu system
    while True:
        print(f"\n=== MENU ===")
        print(f"Current sensitivity: {clicker.THRESHOLD}")
        print("1. Start clicking mode")
        print("2. Test audio levels (recommended first)")
        print("3. Adjust sensitivity")
        print("4. Quit")
        
        choice = input("Choose option (1-4): ").strip()
        
        if choice == "1":
            print("\n=== TIPS ===")
            print("• Make sure your game window is active to receive clicks")
            print("• Start with gentle strumming to test")
            print("• The program will run until you press Ctrl+C")
            print()
            input("Press Enter when ready to start...")
            clicker.start()
            
        elif choice == "2":
            duration = input("Test duration in seconds (default 10): ").strip()
            try:
                duration = int(duration) if duration else 10
            except ValueError:
                duration = 10
            clicker.test_audio_levels(duration)
            
        elif choice == "3":
            try:
                new_threshold = float(input("Enter new sensitivity (0.001-1.0): "))
                clicker.adjust_sensitivity(new_threshold)
            except ValueError:
                print("Invalid input. Please enter a number between 0.001 and 1.0")
                
        elif choice == "4":
            break
        else:
            print("Invalid choice. Please enter 1-4.")
    
    print("Goodbye!")


if __name__ == "__main__":
    main()
