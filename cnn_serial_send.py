import serial
import numpy as np
from serial.tools.list_ports import comports
import struct
from timeit import default_timer as timer
from datetime import timedelta
import timeit
import time
import pickle
import traceback

# X_test = np.load("C:/Users/sanda/Documents/esp_dev_files/data/MNIST_data/MNIST_X_test.npy")
# y_test = np.load("C:/Users/sanda/Documents/esp_dev_files/data/MNIST_data/MNIST_y_test.npy")


X_test = np.load("C:/Users/sanda/Documents/esp_dev_files/data/ecg_specto_data_2/X_test_3.npy")
y_test = np.load("C:/Users/sanda/Documents/esp_dev_files/data/ecg_specto_data_2/y_test_3.npy")

# with open("/Users/sanda/Documents/esp_dev_files/data/ecg_specto_data_2/X_test_2.pcl", "rb") as f:
#     X_test = pickle.load(f)

# with open("/Users/sanda/Documents/esp_dev_files/data/ecg_specto_data_2/y_test_2.pcl", "rb") as f:
#     y_test = pickle.load(f)


# set parameters for serial port
portName='COM6'
#portName = 'COM3' 115200
baudRate = 921600 

# attempt to open port
try:
    ser = serial.Serial()
    ser.port = portName
    ser.baudrate = baudRate
    ser.open()
    print("Opening port " + ser.name)

# if fail, print a helpful message
except:
    print("Couldn't open port. Try changing portName to one of the options below:")
    ports_list = comports()
    for port_candidate in ports_list:
        print(port_candidate.device)
    exit(-1)

if ser.is_open == True:
    print("Success!")

else:
    print("Unable to open port :(")
    exit(-1)
    
#float64 double test ()
#revere the byte order in cpp
send_values = 3.0012



# wait until we have an '!' from the Arduino
bytes = ser.readline()

#decode byte
received = bytes.decode('utf-8')
received = received.replace('\r', '').replace('\n', '')
print(received)
if '!' in received is False:
    print("Invalid request received. Continue to wait.")
    exit(-1)


start_time = timer()


ret_responses = []

# with open("/Users/sanda/Documents/esp_dev_files/tensor_4_ml_2/src/ret_responses_2.pcl", "rb") as f:
#         ret_responses = pickle.load(f)
# send = np.array([1.111,2.222,3.333,4.444])

try: 
    for i in range(int(len(X_test) / 2) , len(X_test)):

        send_values = X_test[i].flatten()
        curr_time = timer()
        print("Send start Time: ", timedelta(seconds=curr_time - start_time))

        for j in range(len(send_values)):
            as_short = float(send_values[j])
            float_format = ">f"

            float_bytes =  bytearray(struct.pack(float_format, as_short))
            curr_time = timer()
            
            # clear the input buffer in case we've been spammed
            ser.reset_input_buffer()

            # send bytes down to Arduino
            ser.write(float_bytes)
            # print("af wr")
            while True:
                # print("in lp")
                bytes = ser.readline()
                
                # decode bytes into string
                response = bytes.decode('utf-8')
                response = response.replace('\r', '').replace('\n', '')
                # print(response)
                curr_time = timer()
                
                # print(timedelta(seconds=curr_time - start_time) , ": Target reported: " , response)

                if ("!" in response or "$" in response):
                    break
                # time.sleep(0.001)
        curr_time = timer()
        print("Send end Time: ", timedelta(seconds=curr_time - start_time))
        # print("Expected Total: ", np.sum(send_values))
        # print("Expected Prediction: ", y_test[i])
        end_send = timedelta(seconds=curr_time - start_time)
        while True:
            
            bytes = ser.readline()

            # decode bytes into string
            response = bytes.decode('utf-8')
            response = response.replace('\r', '').replace('\n', '')
            curr_time = timer()
            # print(response)
            # print(timedelta(seconds=curr_time - start_time) , ": Target reported: " , response)
            if "@" in response:
                end_inference = timedelta(seconds=curr_time - start_time)
                print("ret res ", y_test[i].argmax(), int(response.split(" ")[1]))
                ret_responses.append([end_send, end_inference, y_test[i].argmax(), int(response.split(" ")[1])])
            if ("%" in response):
                break
except Exception as e:
    # print(Exception)
    traceback.print_exc()
    print(ret_responses)
    with open("/Users/sanda/Documents/esp_dev_files/tensor_4_ml/src/ret_responses_3.pcl", "wb") as f:
        pickle.dump(ret_responses, f)
# while(True):
#     # wait until we have an '!' from the Arduino
#     bytes = ser.readline()

#     # decode bytes into string
#     received = bytes.decode('utf-8')
#     received = received.replace('\r', '').replace('\n', '')
#     print(received)
#     break
        # time.sleep(0.0001)
print("--------------------------------")
print(ret_responses)
with open("/Users/sanda/Documents/esp_dev_files/tensor_4_ml/src/ret_responses_3.pcl", "wb") as f:
    pickle.dump(ret_responses, f)
end_time = timer()
print(f'main done in: {timedelta(seconds=end_time- start_time)}\n')

