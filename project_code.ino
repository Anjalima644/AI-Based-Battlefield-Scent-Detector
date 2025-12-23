#define FLATBUFFERS_SPAN_MINIMAL   // must be first

#include <WiFi.h>
#include <ThingSpeak.h>
#include <TensorFlowLite_ESP32.h>  // make sure this is the only TFLM library

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

// ----- Arena -----
constexpr int kTensorArenaSize = 60 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

// ----- Labels -----
const char* labels[] = {
  "LPG",
  "Alcohol",
  "Ammonia",
  "Benzen",
  "Methanol",
  "Normal"
};

// ----- WiFi / ThingSpeak -----
const char* ssid = "Bhavana";
const char* password = "Bhavana12";

unsigned long channelID = 3184037;
const char* writeAPIKey = "99XX5X4IJIRURII6";
WiFiClient client;

// ----- Sensors -----
const int mq2_pin   = 34;
const int mq3_pin   = 35;
const int mq135_pin = 32;

// ----- TFLite objects -----
static tflite::MicroErrorReporter micro_error_reporter;
static tflite::AllOpsResolver resolver;
static tflite::MicroInterpreter* interpreter = nullptr;
static const tflite::Model* model = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

float normalize(float x) {
  return x / 4095.0f;
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  // WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected!");
  ThingSpeak.begin(client);

  // Load model (use the array name from model_data.h)
  model = tflite::GetModel(gas_classifier_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1);
  }

  // Interpreter
  static tflite::MicroInterpreter static_interpreter(
      model,
      resolver,
      tensor_arena,
      kTensorArenaSize,
      &micro_error_reporter);

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation FAILED!");
    while (1);
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Setup complete.");
}

void loop() {
  float mq2_raw   = analogRead(mq2_pin);
  float mq3_raw   = analogRead(mq3_pin);
  float mq135_raw = analogRead(mq135_pin);

  Serial.println("---- Sensor Values ----");
  Serial.println(mq2_raw);
  Serial.println(mq3_raw);
  Serial.println(mq135_raw);

  // Prepare input
  input->data.f[0] = normalize(mq2_raw);
  input->data.f[1] = normalize(mq3_raw);
  input->data.f[2] = normalize(mq135_raw);

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Model inference FAILED!");
    delay(5000);
    return;
  }

  // ----- DEBUG: inspect output tensor -----
  Serial.print("Output type: ");
  Serial.println(output->type);

  Serial.print("Output dims: ");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();

  int numClasses = output->dims->data[output->dims->size - 1];
  Serial.print("numClasses: ");
  Serial.println(numClasses);

  Serial.println("Raw output values:");
  for (int i = 0; i < numClasses; i++) {
    Serial.print(i);
    Serial.print(": ");
    Serial.println(output->data.f[i], 6);
  }

  // ----- Argmax over all classes -----
  int   maxIndex = 0;
  float maxValue = output->data.f[0];
  for (int i = 1; i < numClasses; i++) {
    if (output->data.f[i] > maxValue) {
      maxValue = output->data.f[i];
      maxIndex = i;
    }
  }

  String detectedGas = labels[maxIndex];
  Serial.print("Predicted Gas: ");
  Serial.println(detectedGas);
  Serial.print("Class index: ");
  Serial.println(maxIndex);
  Serial.print("Confidence: ");
  Serial.println(maxValue, 6);

  // ----- Send to ThingSpeak -----
  ThingSpeak.setField(1, mq2_raw);
  ThingSpeak.setField(2, mq3_raw);
  ThingSpeak.setField(3, mq135_raw);
  ThingSpeak.setField(4, (float)maxIndex);  // class index
  ThingSpeak.setField(5, maxValue);         // probability / score

  int response = ThingSpeak.writeFields(channelID, writeAPIKey);
  if (response == 200) {
    Serial.println("ThingSpeak Update Successful!");
  } else {
    Serial.print("ThingSpeak Update Failed: ");
    Serial.println(response);
  }

  Serial.println("------------------------");
  delay(15000);
}