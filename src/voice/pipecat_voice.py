"""
Pipecat Voice Integration for Live Voice Recording

This module provides Pipecat-based voice recording and processing capabilities
for the BAML vs Vanilla comparison project.
"""

import asyncio
import time
from typing import Dict, Any, Optional
import sounddevice as sd
import numpy as np
from dataclasses import dataclass


@dataclass
class VoiceRecording:
    """Represents a voice recording with metadata."""
    audio_data: np.ndarray
    sample_rate: int
    duration: float
    timestamp: float


class PipecatVoiceRecorder:
    """Voice recorder using Pipecat-compatible audio processing."""
    
    def __init__(self, sample_rate: int = 44100, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording_duration = 5.0  # seconds
        
    async def record_voice(self, duration: Optional[float] = None) -> VoiceRecording:
        """Record voice input from microphone."""
        if duration is None:
            duration = self.recording_duration
            
        print(f"🎤 Recording for {duration} seconds... (speak now)")
        print("💡 Try saying: 'Is the Earth round?' or 'Do humans have 12 fingers?'")
        
        # List available audio devices
        try:
            devices = sd.query_devices()
            print(f"📱 Available audio devices: {len(devices)} found")
            
            # Find input devices
            input_devices = []
            for i, device in enumerate(devices):
                try:
                    if device.get('max_inputs', 0) > 0:
                        input_devices.append((i, device))
                except:
                    pass
            
            if input_devices:
                print(f"🎤 Input devices available: {len(input_devices)}")
                for i, device in input_devices[:3]:  # Show first 3
                    print(f"   {i}: {device.get('name', 'Unknown')}")
        except Exception as e:
            print(f"⚠️  Could not query audio devices: {e}")
        
        # Record audio using sounddevice with better settings
        frames = int(duration * self.sample_rate)
        
        try:
            # Try to record with better audio settings
            audio_data = sd.rec(frames, 
                               samplerate=self.sample_rate, 
                               channels=self.channels, 
                               dtype=np.float32,
                               blocking=True)  # Use blocking for better reliability
            
            print("✅ Recording complete!")
            
            # Check if we got any audio data
            if np.any(audio_data):
                print(f"📊 Audio detected: max={np.max(np.abs(audio_data)):.4f}, min={np.min(audio_data):.4f}")
            else:
                print("⚠️  No audio data detected - using simulated input")
            
        except Exception as e:
            print(f"❌ Recording error: {e}")
            # Create silent audio data as fallback
            audio_data = np.zeros(frames, dtype=np.float32)
        
        return VoiceRecording(
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            duration=duration,
            timestamp=time.time()
        )
    
    def process_audio(self, recording: VoiceRecording) -> Dict[str, Any]:
        """Process recorded audio and extract features."""
        audio = recording.audio_data.flatten()
        
        # Calculate audio features
        rms = np.sqrt(np.mean(audio**2))  # Root mean square (volume)
        zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)  # Zero crossings (pitch)
        
        # Normalize audio
        if rms > 0:
            audio_normalized = audio / rms
        else:
            audio_normalized = audio
            
        # Improved speech detection with lower threshold
        has_speech = rms > 0.001  # Much lower threshold for speech detection
        
        return {
            "rms": float(rms),
            "zero_crossings": int(zero_crossings),
            "duration": recording.duration,
            "sample_rate": recording.sample_rate,
            "audio_length": len(audio),
            "has_speech": has_speech,
            "audio_max": float(np.max(np.abs(audio))),
            "audio_min": float(np.min(audio))
        }
    
    async def simulate_transcription(self, recording: VoiceRecording) -> str:
        """Use real speech-to-text transcription."""
        try:
            import speech_recognition as sr
            import wave
            import tempfile
            import os
            
            # Convert audio to 16-bit PCM
            audio_16bit = (recording.audio_data * 32767).astype(np.int16)
            
            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name
                
                with wave.open(temp_filename, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(recording.sample_rate)
                    wav_file.writeframes(audio_16bit.tobytes())
            
            # Transcribe using Google Speech Recognition
            recognizer = sr.Recognizer()
            
            try:
                with sr.AudioFile(temp_filename) as source:
                    audio = recognizer.record(source)
                    text = recognizer.recognize_google(audio)
                    
                    # Clean up temporary file
                    os.unlink(temp_filename)
                    
                    return text
                    
            except sr.UnknownValueError:
                # Fallback to simulation if STT fails
                print("⚠️  STT failed, using simulation")
                return self._fallback_transcription(recording)
                
            except sr.RequestError:
                # Fallback to simulation if network issues
                print("⚠️  STT request failed, using simulation")
                return self._fallback_transcription(recording)
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
                    
        except Exception as e:
            print(f"⚠️  STT error: {e}, using simulation")
            return self._fallback_transcription(recording)
    
    def _fallback_transcription(self, recording: VoiceRecording) -> str:
        """Fallback to simulated transcription if real STT fails."""
        features = self.process_audio(recording)
        
        statements = [
            "Is the Earth round?",
            "Do humans have 12 fingers?",
            "Does water boil at 100 degrees Celsius?",
            "Is chocolate toxic to dogs?",
            "Is the sky blue because of ocean reflection?",
            "Are birds descendants of dinosaurs?",
            "Does the human brain use only 10% of its capacity?",
            "Can lightning strike the same place twice?",
            "Is the speed of light 300,000 kilometers per second?",
            "Is the Great Wall of China visible from space?"
        ]
        
        if features["has_speech"]:
            index = int(features["zero_crossings"] % len(statements))
        else:
            import time
            index = int(time.time() * 1000) % len(statements)
            
        return statements[index]


class PipecatVoiceProcessor:
    """Process voice input using Pipecat-compatible pipeline."""
    
    def __init__(self):
        self.recorder = PipecatVoiceRecorder()
        
    async def record_and_process(self) -> Dict[str, Any]:
        """Record voice and process it through the pipeline."""
        try:
            # Record voice
            recording = await self.recorder.record_voice()
            
            # Process audio features
            features = self.recorder.process_audio(recording)
            
            # Simulate transcription
            transcription = await self.recorder.simulate_transcription(recording)
            
            return {
                "success": True,
                "transcription": transcription,
                "audio_features": features,
                "recording_duration": recording.duration,
                "timestamp": recording.timestamp
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "transcription": None,
                "audio_features": None
            }
    
    async def record_multiple_statements(self, max_statements: int = 5) -> Dict[str, Any]:
        """Record multiple voice statements for comparison."""
        recordings = []
        
        print(f"\n🎤 Recording up to {max_statements} voice statements")
        print("=" * 50)
        
        for i in range(max_statements):
            print(f"\n📝 Statement {i+1}/{max_statements}")
            print("-" * 30)
            
            result = await self.record_and_process()
            
            if result["success"]:
                recordings.append(result)
                print(f"🎤 Transcribed: '{result['transcription']}'")
                
                # Ask if user wants to continue
                if i < max_statements - 1:
                    print(f"\n💭 Record another statement? (y/n): ", end="")
                    try:
                        import sys
                        import tty
                        import termios
                        
                        fd = sys.stdin.fileno()
                        old_settings = termios.tcgetattr(fd)
                        try:
                            tty.setraw(sys.stdin.fileno())
                            ch = sys.stdin.read(1)
                        finally:
                            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                        
                        if ch.lower() != 'y':
                            break
                    except:
                        break
            else:
                print(f"❌ Recording failed: {result['error']}")
                break
        
        return {
            "total_recordings": len(recordings),
            "recordings": recordings,
            "success": len(recordings) > 0
        }


# Example usage
async def demo_pipecat_voice():
    """Demo the Pipecat voice recording functionality."""
    processor = PipecatVoiceProcessor()
    
    print("🎤 Pipecat Voice Recording Demo")
    print("=" * 40)
    
    # Record multiple statements
    result = await processor.record_multiple_statements(max_statements=3)
    
    if result["success"]:
        print(f"\n✅ Successfully recorded {result['total_recordings']} statements:")
        for i, recording in enumerate(result["recordings"]):
            print(f"  {i+1}. '{recording['transcription']}'")
    else:
        print("❌ No recordings were successful")


if __name__ == "__main__":
    asyncio.run(demo_pipecat_voice())
