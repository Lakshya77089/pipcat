# Project Structure

## 📁 Core Files

### Main Scripts
- `voice_comparison.py` - Main Pipecat voice comparison with live recording
- `pipecat_voice_demo.py` - Pipecat voice recording demo

### Configuration
- `requirements.txt` - Full dependencies including voice support
- `requirements-noaudio.txt` - Text-only dependencies
- `env.example` - Environment variables template
- `pyproject.toml` - Project configuration
- `.gitignore` - Git ignore rules

### Documentation
- `README.md` - Main project documentation
- `SUBMISSION_SUMMARY.md` - Project submission summary
- `PROJECT_STRUCTURE.md` - This file

## 📁 Source Code (`src/`)

### Agents
- `src/agents/vanilla_agent.py` - Vanilla prompting agent
- `src/agents/baml_agent.py` - BAML-structured agent

### Voice Processing
- `src/voice/voice_demo.py` - Voice comparison demo
- `src/voice/pipecat_voice.py` - Pipecat voice integration

### Comparison & Metrics
- `src/comparison/runner.py` - Comparison orchestration
- `src/metrics/` - Performance metrics collection
- `src/utils/` - Utility functions

### Configuration
- `src/config/settings.py` - Application settings

## 📁 Test Data (`tests/`)

- `tests/test_data.py` - Test statements for fact-checking
- `tests/test_*.py` - Unit tests for components

## 📁 Documentation (`docs/`)

- Analysis reports and documentation

## 📁 Results

- `comparison_results/` - Generated comparison results (cleaned)
- `voice_samples/` - Sample voice interactions
- `metrics/` - Collected performance metrics

## 🧹 Simplified Project

The following files have been removed to focus on Pipecat voice functionality:
- `run_comparison.py` - Text-only comparison (removed)
- `demo_live_voice.py` - Voice demo (removed)
- `demo_final.py` - Final demo (removed)
- `demo_side_by_side.py` - Side-by-side demo (removed)
- `final_voice_test.py` - Test file (removed)
- `simple_stt_test.py` - Test file (removed)
- `real_stt_test.py` - Test file (removed)
- `real_voice_test.py` - Test file (removed)
- `listen_test.py` - Test file (removed)
- `test_microphone.py` - Test file (removed)
- `temp_audio.wav` - Temporary file (removed)

## 🚀 Quick Commands

```bash
# Main Pipecat voice comparison with live recording
python voice_comparison.py

# Pipecat voice demo
python pipecat_voice_demo.py
```
