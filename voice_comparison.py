#!/usr/bin/env python3
"""
Pipecat Voice + BAML vs. Vanilla Comparison

This is the main entry point for Pipecat voice-enabled comparison between BAML and Vanilla agents.
Includes live voice recording, real-time processing, and comprehensive analysis.
"""

import asyncio
import sys
import os
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

# Ensure the project root is in the Python path for module discovery.
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.voice.voice_demo import VoiceComparisonDemo
from src.comparison.runner import ComparisonRunner
from tests.test_data import TestData


class VoiceComparisonRunner:
    """
    Voice-enabled comparison runner that includes live voice recording.
    
    This class provides both text-based and voice-based comparisons,
    allowing users to test both approaches with live voice input.
    """
    
    def __init__(self):
        self.text_runner = ComparisonRunner()
        self.voice_runner = VoiceComparisonDemo()
        self.voice_enabled = self._check_voice_dependencies()
    
    def _check_voice_dependencies(self) -> bool:
        """Check if voice recording dependencies are available."""
        try:
            import sounddevice
            return True
        except ImportError:
            print("⚠️  Voice recording dependencies not available.")
            print("   Install with: pip install sounddevice")
            return False
    
    async def run_text_comparison(self, test_count: int = 10) -> Dict[str, Any]:
        """Run the standard text-based comparison."""
        print("📝 Running text-based comparison...")
        return await self.text_runner.run_comparison(test_count=test_count)
    
    async def run_voice_comparison(self, test_count: int = 5) -> Dict[str, Any]:
        """Run voice-based comparison with live recording."""
        if not self.voice_enabled:
            print("❌ Voice recording not available. Running text comparison instead.")
            return await self.run_text_comparison(test_count)
        
        print("🎤 Running voice-based comparison...")
        
        # Create test statements for voice comparison
        test_statements = [
            "Is the Earth round?",
            "Do humans have 12 fingers?",
            "Does water boil at 100 degrees Celsius?",
            "Is chocolate toxic to dogs?",
            "Is the sky blue because of ocean reflection?"
        ]
        
        # Limit to requested count
        test_statements = test_statements[:test_count]
        
        return await self.voice_runner.run_voice_comparison(test_statements)
    
    async def run_live_voice_test(self) -> Dict[str, Any]:
        """Run a live voice test where user speaks and both agents respond."""
        if not self.voice_enabled:
            print("❌ Voice recording not available.")
            return None
        
        print("\n🎤 LIVE VOICE TEST")
        print("=" * 50)
        print("Speak a fact-checking statement and both agents will respond.")
        print("Press Ctrl+C to exit.")
        print("=" * 50)
        
        try:
            return await self.voice_runner.run_live_voice_test()
        except KeyboardInterrupt:
            print("\n⏹️  Live voice test interrupted.")
            return None
    
    async def run_comprehensive_comparison(self, 
                                         text_test_count: int = 10,
                                         voice_test_count: int = 5,
                                         include_live_voice: bool = True) -> Dict[str, Any]:
        """Run a comprehensive comparison including both text and voice tests."""
        
        print("🚀 COMPREHENSIVE PIPEchat + BAML vs. VANILLA COMPARISON")
        print("=" * 70)
        print(f"Text tests: {text_test_count} statements")
        print(f"Voice tests: {voice_test_count} statements")
        print(f"Live voice: {'Yes' if include_live_voice and self.voice_enabled else 'No'}")
        print("=" * 70)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "text_comparison": None,
            "voice_comparison": None,
            "live_voice_test": None,
            "overall_winner": None
        }
        
        # Run text comparison
        print("\n📝 PHASE 1: Text-based Comparison")
        print("-" * 40)
        results["text_comparison"] = await self.run_text_comparison(text_test_count)
        
        # Run voice comparison
        if self.voice_enabled:
            print("\n🎤 PHASE 2: Voice-based Comparison")
            print("-" * 40)
            results["voice_comparison"] = await self.run_voice_comparison(voice_test_count)
        
        # Run live voice test
        if include_live_voice and self.voice_enabled:
            print("\n🎤 PHASE 3: Live Voice Test")
            print("-" * 40)
            try:
                results["live_voice_test"] = await self.run_live_voice_test()
            except Exception as e:
                print(f"⚠️  Live voice test failed: {e}")
                results["live_voice_test"] = None
        
        # Determine overall winner
        results["overall_winner"] = self._determine_overall_winner(results)
        
        # Print final results
        self._print_comprehensive_results(results)
        
        return results
    
    def _determine_overall_winner(self, results: Dict[str, Any]) -> str:
        """Determine the overall winner based on all comparison results."""
        
        scores = {"vanilla": 0, "baml": 0}
        
        # Text comparison scoring
        if results["text_comparison"]:
            text_winner = results["text_comparison"]["comparison"]["winner"]
            if text_winner == "BAML":
                scores["baml"] += 2
            elif text_winner == "Vanilla":
                scores["vanilla"] += 2
        
        # Voice comparison scoring
        if results["voice_comparison"]:
            try:
                voice_winner = results["voice_comparison"]["comparison"]["winner"]
                if voice_winner == "BAML":
                    scores["baml"] += 3  # Voice gets higher weight
                elif voice_winner == "Vanilla":
                    scores["vanilla"] += 3
            except (KeyError, TypeError):
                # If voice comparison failed, skip scoring
                pass
        
        # Live voice test scoring
        if results["live_voice_test"]:
            live_winner = results["live_voice_test"].get("winner")
            if live_winner == "BAML":
                scores["baml"] += 2
            elif live_winner == "Vanilla":
                scores["vanilla"] += 2
        
        # Determine winner
        if scores["baml"] > scores["vanilla"]:
            return "BAML"
        elif scores["vanilla"] > scores["baml"]:
            return "Vanilla"
        else:
            return "Tie"
    
    def _print_comprehensive_results(self, results: Dict[str, Any]):
        """Print comprehensive comparison results."""
        
        print("\n" + "=" * 70)
        print("🏆 COMPREHENSIVE COMPARISON RESULTS")
        print("=" * 70)
        
        # Text comparison results
        if results["text_comparison"]:
            text_comp = results["text_comparison"]["comparison"]
            print(f"\n📝 Text Comparison Winner: {text_comp['winner']}")
            print(f"   Vanilla Accuracy: {text_comp['performance_metrics']['vanilla']['accuracy_rate']:.1%}")
            print(f"   BAML Accuracy: {text_comp['performance_metrics']['baml']['accuracy_rate']:.1%}")
        
        # Voice comparison results
        if results["voice_comparison"]:
            voice_comp = results["voice_comparison"]
            print(f"\n🎤 Voice Comparison Winner: {voice_comp['comparison']['winner']}")
            print(f"   Vanilla Turn Accuracy: {voice_comp['comparison']['voice_metrics']['turn_accuracy']['vanilla']:.1%}")
            print(f"   BAML Turn Accuracy: {voice_comp['comparison']['voice_metrics']['turn_accuracy']['baml']:.1%}")
        
        # Live voice test results
        if results["live_voice_test"]:
            live_test = results["live_voice_test"]
            print(f"\n🎤 Live Voice Test Winner: {live_test.get('winner', 'N/A')}")
            print(f"   User Satisfaction: {live_test.get('user_satisfaction', 'N/A')}")
        
        # Overall winner
        print(f"\n🏆 OVERALL WINNER: {results['overall_winner']}")
        print("=" * 70)
        
        # Detailed analysis
        print("\n📊 DETAILED ANALYSIS:")
        if results["overall_winner"] == "BAML":
            print("✅ BAML wins due to:")
            print("   • Structured prompt definitions")
            print("   • Better voice interaction handling")
            print("   • Enhanced confidence scoring")
            print("   • Improved conversational flow")
            print("   • Type-safe response validation")
        elif results["overall_winner"] == "Vanilla":
            print("✅ Vanilla wins due to:")
            print("   • Simpler implementation")
            print("   • Lower latency in some cases")
            print("   • Direct prompt control")
        else:
            print("🤝 Tie - Both approaches perform similarly")


async def main():
    """Main function to run the voice comparison."""
    
    print("🎤 Pipecat Voice + BAML vs. Vanilla Comparison")
    print("=" * 60)
    
    runner = VoiceComparisonRunner()
    
    # Check available options
    print("\nAvailable comparison modes:")
    print("1. Text-only comparison")
    print("2. Voice-only comparison")
    print("3. Live voice test")
    print("4. Comprehensive comparison (all modes)")
    
    try:
        choice = input("\nSelect mode (1-4): ").strip()
        
        if choice == "1":
            results = await runner.run_text_comparison()
        elif choice == "2":
            results = await runner.run_voice_comparison()
        elif choice == "3":
            results = await runner.run_live_voice_test()
        elif choice == "4":
            results = await runner.run_comprehensive_comparison()
        else:
            print("Invalid choice. Running comprehensive comparison...")
            results = await runner.run_comprehensive_comparison()
        
        if results:
            print(f"\n🎉 Comparison completed successfully!")
            print(f"📁 Results saved to comparison_results/ directory")
        else:
            print(f"\n⚠️  Comparison completed with issues.")
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Comparison interrupted by user.")
    except Exception as e:
        print(f"\n💥 Error during comparison: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n⏹️  Program interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
