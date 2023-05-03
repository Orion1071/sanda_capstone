# Sanda Capstone
# Analysis of the Machine Learning Classification of Cardiac Disease on Embedded Systems
## Paper can be found [here](https://www.jasonforsyth.net/pdf/thursa-sieds-2023.pdf)
---

# Instruction for implementation

- Train the model
- Convert the model appropirately accordinging to your requriments. Example model given in ``cnn_ecg_10``
- Using ``xxd -i <tflite_file_name>`` to convert to C++ byte array. Example ``xxd -i cnn_ecg_10.tflite > cnn_ecg_10.cc``
- Open the byte array file and convert that array into ``const``. Add const in front of the array type. This is required by tensorflowlite guideline to save memory if you are working with big models. 
- Import that byte array file in the main.cpp file. Example file given in main.cpp
- Using the platformIO, build and port onto the device using the instruction. 
- load the tensorflow-micro framework from tensorflow website. If you are having compiler issues, you can safely delete some experimental files from the framework.
- If you need to enable external ram, type in ``pio run -t menuconfig``. find your external RAM in that menu. enable it. 
- If the model will run too long, you may also need to diable watchdog in the same menuconfig
- For online processing, use the ``serail_send.py``
- make changes to the datapath appropirately
- Do no hesitate to contact me. The contact information can be found on [sanda](sanda.thura.me)
