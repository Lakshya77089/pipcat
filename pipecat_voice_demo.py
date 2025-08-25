#!/usr/bin/env python3
"""
Pipecat Voice Recording Demo

This script demonstrates live voice recording using Pipecat-compatible
audio processing for the BAML vs Vanilla comparison project.
"""

import asyncio
import sys
from src.voice.pipecat_voice import PipecatVoiceProcessor
from src.voice.voice_demo import VoiceComparisonDemo


async def demo_pipecat_voice_recording():
    """Demo the Pipecat voice recording functionality."""
    print("🎤 Pipecat Voice Recording Demo")
    print("=" * 50)
    print("This demo shows live voice recording with Pipecat integration")
    print("and compares BAML vs Vanilla agents on voice input.")
    print("=" * 50)
    
    # Initialize Pipecat voice processor
    processor = PipecatVoiceProcessor()
    
    # Record multiple statements
    print("\n📝 Recording voice statements for comparison...")
    result = await processor.record_multiple_statements(max_statements=3)
    
    if result["success"]:
        print(f"\n✅ Successfully recorded {result['total_recordings']} statements:")
        for i, recording in enumerate(result["recordings"]):
            print(f"  {i+1}. '{recording['transcription']}'")
        
        # Now run the voice comparison with the recorded statements
        print("\n🔄 Running voice comparison with recorded statements...")
        
        voice_demo = VoiceComparisonDemo()
        test_statements = [r["transcription"] for r in result["recordings"]]
        
        comparison_result = await voice_demo.run_voice_comparison(test_statements)
        
        print("\n🎉 Voice comparison completed!")
        print(f"🏆 Winner: {comparison_result['comparison']['winner']}")
        
    else:
        print("❌ No recordings were successful")


async def demo_live_voice_comparison():
    """Demo the live voice comparison functionality."""
    print("🎤 Live Voice Comparison Demo")
    print("=" * 50)
    print("This demo shows live voice recording and real-time comparison")
    print("between BAML and Vanilla agents.")
    print("=" * 50)
    
    voice_demo = VoiceComparisonDemo()
    
    if not voice_demo.voice_enabled:
        print("❌ Voice recording not available. Install sounddevice.")
        return
    
    # Run live voice test
    result = await voice_demo.run_live_voice_test()
    
    if result:
        print(f"\n🎉 Live voice test completed!")
        print(f"📊 Total statements: {result['total_statements']}")
        print(f"🏆 Overall Winner: {result['winner']}")
        print(f"   Vanilla wins: {result['vanilla_wins']}")
        print(f"   BAML wins: {result['baml_wins']}")
    else:
        print("❌ Live voice test failed or was interrupted")


def main():
    """Main function with menu selection."""
    print("🎤 Pipecat Voice Demo Menu")
    print("=" * 30)
    print("1. Pipecat voice recording demo")
    print("2. Live voice comparison demo")
    print("3. Both demos")
    print("4. Exit")
    
    try:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            asyncio.run(demo_pipecat_voice_recording())
        elif choice == "2":
            asyncio.run(demo_live_voice_comparison())
        elif choice == "3":
            print("\n" + "="*50)
            asyncio.run(demo_pipecat_voice_recording())
            print("\n" + "="*50)
            asyncio.run(demo_live_voice_comparison())
        elif choice == "4":
            print("👋 Goodbye!")
            sys.exit(0)
        else:
            print("❌ Invalid choice. Please select 1-4.")
            
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted.")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
