/*
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Recognizes various sounds heard with the on-board mic, using YamNet,
// running on the Edge TPU (though it can be modified to run on the CPU).
// See model details here: https://tfhub.dev/google/yamnet/1

#include <PDM.h>
#include <coralmicro_SD.h>

#include "Arduino.h"
#include "coral_micro.h"
#include "libs/tensorflow/audio_models.h"
#include "libs/tensorflow/classification.h"
#include "third_party/tflite-micro/tensorflow/lite/experimental/microfrontend/lib/frontend.h"

// From Blink without Delay.
// constants won't change. Used here to set a pin number:
const int ledPin = LED_BUILTIN;  // the number of the LED pin

// Pins for Audio Playback
const int playbackPin0 = A1;
//const int playbackPin1 = A0;
//const int playbackPin2 = D2;
//const int playbackPin3 = D1;

// Variables will change:
int ledState = LOW;  // ledState used to set the LED

// Generally, you should use "unsigned long" for variables that hold time
// The value will quickly become too large for an int to store
unsigned long blink_until = 0;  // timestamp longer to blink (milliseconds)

namespace {
bool setup_success{false};

std::vector<int32_t> current_samples;
tflite::MicroMutableOpResolver</*tOpsCount=*/3> resolver;
const tflite::Model* model = nullptr;
std::vector<uint8_t> model_data;
std::shared_ptr<coralmicro::EdgeTpuContext> context = nullptr;
std::unique_ptr<tflite::MicroInterpreter> interpreter = nullptr;
std::array<int16_t, coralmicro::tensorflow::kYamnetAudioSize> audio_input;

constexpr int kTensorArenaSize = 1 * 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

FrontendState frontend_state{};
constexpr float kThreshold = 0.3;
constexpr int kTopK = 5;

constexpr char kModelName[] = "/yamnet_spectra_in_edgetpu.tflite";
}  // namespace

void setup() {
  Serial.begin(115200);
  //delay(5000);  // Add delay to print if setup doesn't work.
  // Turn on Status LED to show the board is on.
  pinMode(PIN_LED_STATUS, OUTPUT);
  pinMode(ledPin, OUTPUT);  // Blink without delay.
  digitalWrite(ledPin, LOW);
  digitalWrite(PIN_LED_STATUS, HIGH);
  
  // Pin init for audio playback (note that grounding pin turns on audio).
  pinMode(playbackPin0, OUTPUT);
  //pinMode(playbackPin1, OUTPUT);
  //pinMode(playbackPin2, OUTPUT);
  //pinMode(playbackPin3, OUTPUT);
  digitalWrite(playbackPin0, HIGH);
  //digitalWrite(playbackPin1, HIGH);
  //digitalWrite(playbackPin2, HIGH);
  //digitalWrite(playbackPin3, HIGH);

  Serial.println("Arduino YamNet!");

  SD.begin();
  Mic.begin(coralmicro::tensorflow::kYamnetSampleRate,
            coralmicro::tensorflow::kYamnetDurationMs);

  tflite::MicroErrorReporter error_reporter;
  resolver = coralmicro::tensorflow::SetupYamNetResolver</*tForTpu=*/true>();

  Serial.println("Loading Model");

  // Debug for if setup doesnt work.
  printDirectoryTree();

  if (!SD.exists(kModelName)) {
    Serial.println("Model file not found");
    return;
  }

  SDFile model_file = SD.open(kModelName);
  uint32_t model_size = model_file.size();
  model_data.resize(model_size);
  if (model_file.read(model_data.data(), model_size) != model_size) {
    Serial.print("Error loading model");
    return;
  }

  model = tflite::GetModel(model_data.data());
  context = coralmicro::EdgeTpuManager::GetSingleton()->OpenDevice();
  if (!context) {
    Serial.println("Failed to get EdgeTpuContext");
    return;
  }

  interpreter = std::make_unique<tflite::MicroInterpreter>(
      model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed.");
    return;
  }

  if (!coralmicro::tensorflow::PrepareAudioFrontEnd(
          &frontend_state, coralmicro::tensorflow::AudioModel::kYAMNet)) {
    Serial.println("coralmicro::tensorflow::PrepareAudioFrontEnd() failed.");
    return;
  }

  setup_success = true;
  Serial.println("YAMNet Setup Complete\r\n\n");
}

void loop() {
  if (!setup_success) {
    //Serial.println("Cannot invoke because of a problem during setup!");
    return;
  }

  unsigned long currentMillis = millis();
  if (blink_until > currentMillis) {
    if (digitalRead(ledPin) == LOW) {
      digitalWrite(ledPin, HIGH);
      digitalWrite(playbackPin0, LOW);  // Ground pin to play audio.
    }
  } else {
    // Turn off LED and audio pin.
    if (digitalRead(ledPin) == HIGH) {
      digitalWrite(ledPin, LOW);
      // Don't continue audio playback (otherwise it keeps repeating)
      digitalWrite(playbackPin0, HIGH);   // HIGH is off.
    }
  }

  if (Mic.available() < coralmicro::tensorflow::kYamnetAudioSize) {
    return;  // Wait until we have enough data.
  }
  Mic.read(current_samples, coralmicro::tensorflow::kYamnetAudioSize);

  for (int i = 0; i < coralmicro::tensorflow::kYamnetAudioSize; ++i) {
    audio_input[i] = current_samples[i] >> 16;
  }

  coralmicro::tensorflow::YamNetPreprocessInput(
      audio_input.data(), interpreter->input_tensor(0), &frontend_state);
  // Reset frontend state.
  FrontendReset(&frontend_state);
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Failed to invoke");
    return;
  }

  auto results = coralmicro::tensorflow::GetClassificationResults(
      interpreter.get(), kThreshold, kTopK);
  if (results.empty()) {
    Serial.println("No results");
  } else {
    Serial.println("Results");
    for (const auto& c : results) {
      Serial.print(c.id);
      Serial.print(": ");
      Serial.println(c.score);

      // Blink and play audio if Meow or Cat.
      if (c.id==78 || c.id==76 ) {
        blink_until = currentMillis + 1000;
      }

    }
  }
}

void printDirectoryTree() {
  Serial.println("Current directory tree");
  SDFile dir = SD.open("/");
  printDirectory(dir, 0);
  dir.close();
}

void printDirectory(SDFile dir, int numTabs) {
  // Prints a directory and all its subdirs
  while (true) {
    SDFile entry = dir.openNextFile();
    if (!entry) {
      break;
    }
    for (uint8_t i = 0; i < numTabs; i++) {
      Serial.print('\t');
    }
    Serial.print(entry.name());
    if (entry.isDirectory()) {
      Serial.println("/");
      printDirectory(entry, numTabs + 1);
    } else {
      Serial.print("\t\t");
      Serial.println(entry.size());
    }
    entry.close();
  }
}
