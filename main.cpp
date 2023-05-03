#include <Arduino.h>
#include <stdio.h>
#include <string.h>
#include "models/cnn_mnist_model_2.cc"
#include "models/cnn_small_10.cc"
// #include <interpreter.h>
// #include <model_builder.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/core/api/error_reporter.h"

// Static varaibles section
//-----------------------------

// tensorflow stuff

namespace tflite
{
    template <unsigned int tOpCount>
    class MicroMutableOpResolver;
    class ErrorReporter;
    class Model;
    
    class MicroInterpreter;
} struct TfLiteTensor;

tflite::AllOpsResolver *resolver;
tflite::ErrorReporter *error_reporter;
const tflite::Model *model;
tflite::MicroInterpreter *interpreter;
TfLiteTensor *input;
TfLiteTensor *output;
uint8_t *tensor_arena;
const int kArenaSize = 5000000; 
// float ** tensor_in = nullptr;

//serial data stuff
const int dim1 = 570; //(32, 570, 33) (32, 4)
const int dim2 = 33;

// const int dim1 = 28; //(32, 570, 33) (32, 4)
// const int dim2 = 28; 

// const int num_classes = 10;
const int num_classes = 4;
float arr[dim1*dim2];
const int numBytes = 4;

//bench marking
long time_to_receive_word = 0;
long time_to_copy_word = 0;
//-----------------------------


//find the index of highest probability
int index_finder(float * arr) {
    float max = 0.0;
    int index = -1;
    for(int i=0;i<num_classes;i++)
    {
        if(output->data.f[i]>max) {
            max = output->data.f[i];
            index = i;
        }
    }
    return (index);
}

unsigned char* model_array;

void setup() {
  // put your setup code here, to run once:
  //initialize tensorflow model
  error_reporter = new tflite::MicroErrorReporter();
  // model = tflite::GetModel(cnn_mnist_model_2_tflite);

  // model_array = (uint8_t*)malloc(464041);
  // memcpy(model_array, keras_ecg_cnn_small_4_tflite, 464040);
  // Serial.println((int)keras_ecg_cnn_small_2_tflite);
  model = tflite::GetModel(keras_ecg_cnn_small_10_tflite);
  resolver = new tflite::AllOpsResolver();
  tensor_arena =  (uint8_t *)malloc(kArenaSize);
  
  if (!tensor_arena)
  {
      TF_LITE_REPORT_ERROR(error_reporter, "Could not allocate arena");
      return;
  }
  
  // Build an interpreter to run the model with.
  interpreter = new tflite::MicroInterpreter(
      model, *resolver, tensor_arena, kArenaSize);
  // Allocate the tensor
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
      TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
      return;
  }
  size_t used_bytes = interpreter->arena_used_bytes();
  
  //initialize the board and the serial port
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  /**/
  Serial.begin(921600);
}

void loop() {
  int curr_dim = 0;
  byte array[numBytes];

  // spam the PC to say that we want some data
  while (Serial.available() == 0)
  {
    //send '!' up the chain
    Serial.println("!");

    //spam the host until they respond :)
    delay(10);
  }
  // digitalWrite(LED_BUILTIN, HIGH);
  input = interpreter->input(0);
  while (curr_dim < dim1 * dim2) {
    while (Serial.available() == 0)
    {
      //send '!' up the chain
      Serial.println("!");

      //spam the host until they respond :)
      delay(3);
    }
    // turn on the LED to indicate we're waiting on data
    digitalWrite(LED_BUILTIN, HIGH);
    // generate a time stamp when begin to receive word
    // time_to_receive_word = millis();
  
    // wait until we have enough bytes for single word
    while (Serial.available() < numBytes) {}

    // tare time and prepare to report
    // time_to_receive_word-=millis();

    // copy the bytes into an array
    // time_to_copy_word = millis();
    for (int i = numBytes -1 ; i > -1; i--)
    {
      array[i] = Serial.read();
    }

    // now cast the 32 bits into something we want...
    float value = *((float*)(array));
    // arr[curr_dim] = value;
    input->data.f[curr_dim] = value;
    
    // print out received value
    // time_to_copy_word-=millis();
    curr_dim ++;

    // Serial.print("Rx: ");
    // Serial.print(time_to_receive_word);
    // Serial.print("Arrange: ");
    // Serial.println(time_to_copy_word);
    
    // tell the PC I'm ready for another data point
    Serial.println("$");
    digitalWrite(LED_BUILTIN, LOW);
    delay(1);
  }
  // digitalWrite(LED_BUILTIN, LOW);


  // Serial.print("The sum of the received array is: ");
  // float mul = 0.0;
  // for(int f = 0; f < dim1 * dim2; f++) {
  //   mul = mul + arr[f];
  // }
  // Serial.printf("%f %f %f %f\n",arr[0],arr[1],arr[2],arr[3] );
  // Serial.printf("%f\n", mul);

  // set up input data
  // input = interpreter->input(0);
  // for (int i = 0; i < dim1 * dim2; i++) {
  //       input->data.f[i] = arr[i];
  // }
  // Serial.printf("input finished\n");
  // set up output data1
  output = interpreter->output(0);
  // Serial.printf("output finished\n");
  // Run inference, and report any error.
  // delay(1);
  TfLiteStatus invoke_status = interpreter->Invoke(); 
  // Serial.printf("invoke status %s\n", invoke_status != kTfLiteOk ?"true":"false");
  // Serial.printf("invoke finished\n");
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed\n");
    return;
  }
  // Serial.print("  Tensorflow output \n");
  
  Serial.printf("Prediction@: %d\n", index_finder(output->data.f));
  // delay so the light stays on
  // delay(10);
  // digitalWrite(LED_BUILTIN, LOW);
  Serial.println("%");

}

/*
void loop() {
  int curr_dim = 0;
  byte array[numBytes];

  // spam the PC to say that we want some data
  while (Serial.available() == 0)
  {
    //send '!' up the chain
    Serial.println("!");
    // delay(100);
    // Serial.println("model file");
    // Serial.println((unsigned)(&cnn_mnist_model_2_tflite[0]), HEX);
    // delay(1000);
    // delay(100);
    // Serial.println("tensor place");
    // Serial.println((unsigned)(&tensor_arena[0]), HEX);
    // delay(1000);
    // delay(100);
    // Serial.println("loaded model file");
    // Serial.println((unsigned)(&model[0]), HEX);
    // delay(1000);
    // delay(100);
    // Serial.println("interpreter");
    // Serial.println((unsigned)(&interpreter[0]), HEX);
    // delay(1000);
    //spam the host until they respond :)
    delay(10);

  }
  // digitalWrite(LED_BUILTIN, HIGH);
  input = interpreter->input(0);
  while (curr_dim < dim1 * dim2) {
    while (Serial.available() == 0)
    {
      //send '!' up the chain
      Serial.println("!");

      //spam the host until they respond :)
      delay(3);
    }
    // turn on the LED to indicate we're waiting on data
    digitalWrite(LED_BUILTIN, HIGH);
    // generate a time stamp when begin to receive word
    // time_to_receive_word = millis();
  
    // wait until we have enough bytes for single word
    while (Serial.available() < numBytes) {}

    // tare time and prepare to report
    // time_to_receive_word-=millis();

    // copy the bytes into an array
    // time_to_copy_word = millis();
    for (int i = numBytes -1 ; i > -1; i--)
    {
      array[i] = Serial.read();
    }

    // now cast the 32 bits into something we want...
    // float value = *((float*)(array));
    int8_t value = *((int8_t*)(array));
    // arr[curr_dim] = value;
    // input->data.f[curr_dim] = value;
    input->data.int8[curr_dim] = value;
    
    // print out received value
    // time_to_copy_word-=millis();
    curr_dim ++;

    // Serial.print("Rx: ");
    // Serial.print(time_to_receive_word);
    // Serial.print("Arrange: ");
    // Serial.println(time_to_copy_word);
    
    // tell the PC I'm ready for another data point
    Serial.println("$");
    digitalWrite(LED_BUILTIN, LOW);
    delay(1);
  }
  // digitalWrite(LED_BUILTIN, LOW);


  // Serial.print("The sum of the received array is: ");
  // float mul = 0.0;
  // for(int f = 0; f < dim1 * dim2; f++) {
  //   mul = mul + arr[f];
  // }
  // Serial.printf("%f %f %f %f\n",arr[0],arr[1],arr[2],arr[3] );
  // Serial.printf("%f\n", mul);

  // set up input data
  // input = interpreter->input(0);
  // for (int i = 0; i < dim1 * dim2; i++) {
  //       input->data.f[i] = arr[i];
  // }
  // Serial.printf("input finished\n");
  // set up output data1
  output = interpreter->output(0);
  // Serial.printf("output finished\n");
  // Run inference, and report any error.
  // delay(1);
  TfLiteStatus invoke_status = interpreter->Invoke(); 
  // Serial.printf("invoke status %s\n", invoke_status != kTfLiteOk ?"true":"false");
  // Serial.printf("invoke finished\n");
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed\n");
    return;
  }
  // Serial.print("  Tensorflow output \n");
  
  // Serial.printf("Prediction@: %d\n", index_finder(output->data.f));
  Serial.printf("Prediction@: %d\n", index_finder2(output->data.int8));
  // delay so the light stays on
  // delay(10);
  // digitalWrite(LED_BUILTIN, LOW);
  Serial.println("%");

}

*/