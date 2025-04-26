# Meow Detector

Welcome to the Meow Detector project!  
This project uses the **Coral Dev Board Micro** to detect when my cat meows — and then plays a calming voice recording to soothe him.

It uses:
- **TensorFlow Lite** to run a custom audio classification model on the **Edge TPU**
- The board’s **built-in speaker** for audio playback
- The **green User LED** to indicate when a meow is detected

---

## Project Overview

The Meow Detector is described in more detail on my project website:  
👉 [aaronmanprojects.com](https://aaronmanprojects.com)

You can also check out my blog posts for deeper dives:
- **[Meow Detector – Part 1: Detection](https://aaronmanprojects.com/2023/12/31/overview/)**  
  *In this post, I set up the project to light the green User LED when a meow is detected.*

- **[Meow Detector – Part 2: Audio Playback](https://aaronmanprojects.com/2024/01/06/meow-detector-part-2-audio-playback/)**  
  *This post explains how to add audio playback after detecting a meow.*

---

## Features
- Real-time meow detection using a TensorFlow Lite model
- Edge TPU acceleration
- Visual feedback with User LED
- Audio feedback with the built-in speaker
