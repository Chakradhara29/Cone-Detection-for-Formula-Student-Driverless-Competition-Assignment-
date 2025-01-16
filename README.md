# Cone-Detection-for-Formula-Student-Driverless-Competition-Assignment-
## MATLAB and Simulink Challenge project 248
[PROJECT LINK](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/tree/main/projects/Cone%20Detection%20for%20Formula%20Student%20Driverless%20Competition)


Here’s a comprehensive guide to implementing and testing an object detection algorithm using Simulink and the Vehicle Dynamics Blockset™. This guide will take you through the process of using a skidpad test setup to detect cones in a 3D simulation environment.

---

## Objective

To create a Simulink model using the Vehicle Dynamics Blockset™ that simulates a skidpad test environment, incorporates a camera for visual input, and detects cones in real-time at a vehicle speed of 30 km/h. The algorithm will leverage deep learning techniques, such as YOLO v2, to identify cones and evaluate detection accuracy.

---

## Version Used : MATLAB R2023b

# How to run the trained files uploaded accordingly to obtain the output:

Open Matlab R2023b, load all the files you have downloaded to the directory.
Open "vdynblksskidpad", in deep learning block, change the input file path to "yolov2ConeDetector" which was downloaded. Apply and run the simulation.
Once this is loaded, close the videoViewer and the simulation to make changes in the Camera block, Set the "Parent name:SimulinkVehicle3" & "Mounting location: Roll Ball Centre".
In Deep Learning Object block: Configure the **Detection Threshold** to a suitable level (i.e., 0.47 to filter low-confidence detections & set the no. of iterations to 250).
Once all the changes have been made, run the simulation and will give you the output as same as video i uploded: "detectedConesVideo"


Later for constsnt velocity (refer the complete guide i have given below) and add constant block in connection to VelRef.
For Jetson: Use the guide i provided below and use the syntax taught in formula driverless workshop and code is attacted "JETSON". make sure the devices are conneceted and intact and run it.

---

## Steps : (To Train model and to obtain the output)

First download all the neccessary add on : Refer to ADDON file

### 1. Familiarize Yourself with the Skidpad Test Model in R2023b

The **Generate Skidpad Test** model, introduced in R2023b, is a part of the Vehicle Dynamics Blockset™. This model simulates a vehicle’s performance around a circular path (skidpad), a common setup in vehicle dynamics testing. 

The skidpad model includes:
- A **reference path** for the vehicle to follow.
- A **driver model** to control the vehicle.
- A **vehicle dynamics model** that realistically simulates vehicle motion.
- **Visualization aides**, including a 3D environment powered by Unreal Engine, allowing real-time visualization.

### **2. Setting Up the Simulation Environment**

Open the **Generate Skidpad Test** model in Simulink:
   - You can access this example by searching for "Generate Skidpad Test" in the MATLAB Help or by navigating to `Vehicle Dynamics Blockset™ > Examples > Generate Skidpad Test`.

The simulation will include:
- **Skidpad Reference Path**: Defines the path for the vehicle.
- **Driver Commands**: Controls the vehicle to follow the path at a specified speed.
- **Formula Student Vehicle**: Simulates the physical dynamics of the vehicle.
- **3D Visualization**: A photorealistic rendering of the environment using Unreal Engine.

### **3. Adding Sensor Blocks for Camera Output**

To enable object detection, add a virtual camera to the simulation environment. We’ll use the **Simulation 3D Camera** block from the **Automated Driving Toolbox™**.

1. **Open the Library Browser** in Simulink.
2. Navigate to **Automated Driving Toolbox > Sensors**.
3. Drag the **Simulation 3D Camera** block into your model.

#### Configuring the Camera Block:
- Set the **Mount Location** to an appropriate position on the vehicle (e.g., 'mountLoc','Roll bar center' and use simvehcile3) to capture a clear view of the cones.
- **Output**: The camera will output an image of the scene, which will be used as input to the object detection algorithm.

#### Verifying Camera View:
- Run the simulation briefly to ensure the camera captures the skidpad environment correctly.
- Use a **Video Viewer** block to display the camera output, allowing you to verify the camera's perspective and ensure the cones are visible.

### **4. Developing an Object Detection Algorithm for Cone Detection**

To detect cones in the 3D scene, you can use a deep learning model, such as **YOLO v2**, which is effective for real-time object detection.

#### Steps to Develop and Integrate the Detection Algorithm:

1. **Training the Model**:
   - Collect a dataset of cone images from the simulation or other sources, ensuring images include cones from various angles, distances, and lighting conditions.
   - Use a pre-trained YOLO model (such as one pre-trained on general objects) and fine-tune it with cone images, or train a new model entirely on cone images if you have sufficient data.
   - Use MATLAB’s **Deep Learning Toolbox™** to train the model, which supports YOLO-based object detection.

2. **Add the Object Detection Block**:
   - In Simulink, add a **Deep Learning Object Detector** block. This block can directly use the trained YOLO v2 model.
   - Connect the camera output (the image) to the **Deep Learning Object Detector** block.

3. **Set Detection Parameters**:
   - Configure the **Detection Threshold** to a suitable level (i.e., 0.47 to filter low-confidence detections & set the no. of iterations to 250).
   - Set up the block to output bounding boxes and labels for detected objects.

### **5. Configuring Constant Velocity at 30 km/h**

Since the objective is to detect cones when the vehicle moves at a minimum speed of 30 km/h, ensure the vehicle maintains this speed during simulation.

#### How to Set Constant Velocity:
1. Add a **Constant** block with a value of `8.33` (since 30 km/h is approximately 8.33 m/s).
2. Connect this **Constant** block to the **VelRef** input in the **Driver Commands** block, ensuring the vehicle will target and maintain a constant speed of 30 km/h.

Step-by-step guide to modify the **Driver Commands** block to simulate the vehicle at a constant velocity of 30 km/h (8.33 m/s):

### Step 1: Open Your Simulink Model
1. Open the Simulink model that you are working with (`vdynblksskidpad`).

### Step 2: Add a Constant Block
1. In the **Library Browser**, navigate to the **Sources** section.
   - You can find the **Library Browser** on the Simulink toolstrip or by pressing **Ctrl+Shift+L**.
2. In **Sources**, find the **Constant** block.
3. Drag the **Constant** block and drop it into your model near the **Driver Commands** block (in the upper section where **VelRef** is shown).

### Step 3: Set the Constant Value for Velocity
1. Double-click on the **Constant** block to open its parameters.
2. In the **Constant value** field, enter `8.33` (this is 30 km/h converted to meters per second).
3. Click **OK** to close the parameters window.

### Step 4: Connect the Constant Block to VelRef
1. Find the **VelRef** input on the **Driver Commands** block. From the diagram you uploaded, it's one of the first inputs labeled as **VelRef**.
2. Click and drag a line from the output of the **Constant** block to the **VelRef** input of the **Driver Commands** block.
   - This sets the **VelRef** (Velocity Reference) to the constant value of 8.33 m/s (which is 30 km/h) throughout the simulation.

### Step 5: Update and Run the Simulation
1. After connecting the constant block to **VelRef**, go to the **Modeling** tab and click on **Update Model** to ensure all changes are reflected.
   - Alternatively, you can press **Ctrl+D** to update the model.
2. Run your simulation to verify that the vehicle is now moving at a constant speed of 30 km/h.

### Step 6: Test and Verify
1. You can monitor the vehicle's speed by using **Display** or **Scope** blocks connected to the **VelFdbk** (velocity feedback) signal from the **Driver Commands** block. This will show you whether the vehicle is maintaining a constant speed.

### Visual Guide Recap:
- **Constant Block**: Value = `8.33`.
- **VelRef Input**: Located at the **Driver Commands** block.
- **Connection**: Connect the **Constant Block** output to the **VelRef** input of **Driver Commands**.

This will ensure that your vehicle maintains a constant speed of 30 km/h during the simulation.

---

### During the simulation, you should see bounding boxes around detected cones in the video viewer

---

Deploying a trained deep learning network to a Formula Student car involves adapting the model for real-time processing on embedded hardware, such as an NVIDIA Jetson. This process allows the detection algorithm to run autonomously on the vehicle during live track testing. Here’s a step-by-step guide to deploying your network to an NVIDIA Jetson device, focusing on the steps necessary to move from Simulink simulation to on-car deployment.

---

## **Steps for Deployment on an NVIDIA Jetson**

### **1. Prerequisites**

Ensure the following requirements are met before deployment:
- **Trained Model**: A trained cone-detection network (e.g., YOLO v2) compatible with the NVIDIA Jetson hardware.
- **Hardware Setup**: An NVIDIA Jetson board (e.g., Jetson TX2, Xavier) with the necessary connections and power setup on the Formula Student car.
- **Software**: MathWorks tools such as **MATLAB**, **Simulink**, **Deep Learning Toolbox™, GPU Coder™**, and the **GPU Coder Support Package for NVIDIA GPUs**.

### **2. Install GPU Coder Support Package for NVIDIA GPUs**

The **GPU Coder Support Package for NVIDIA GPUs** provides the tools and drivers required to deploy models directly from MATLAB/Simulink to NVIDIA Jetson boards.

### **3. Configure the Deep Learning Model for Deployment**

The next step is to convert your trained model into a format suitable for deployment on an embedded GPU. GPU Coder will help you do this by generating optimized CUDA code.

1. **Open MATLAB** and load your trained YOLO v2 model.
2. Use **GPU Coder** to convert the deep learning model to CUDA code. This will optimize the model for real-time performance on the NVIDIA Jetson.
   - Example command:
     ```matlab
     cfg = coder.gpuConfig('dll');
     cfg.TargetLang = 'C++';
     cfg.DeepLearningConfig = coder.DeepLearningConfig('cudnn');
     codegen -config cfg detectCones -args {inputSize} -report
     ```
   - Replace `detectCones` with your detection function and `inputSize` with the appropriate input dimensions for your model.

3. **Transfer the Model to Simulink**: After generating CUDA code, integrate the code with Simulink using the **MATLAB Function block** to call the detection function. This function should receive camera input and output bounding boxes and labels.

### **4. Deploy the Model to the NVIDIA Jetson Board**

Once the CUDA-optimized model is ready, set up your Simulink model to deploy to the Jetson hardware.

1. **Open the Simulink Model**: Open the skidpad model and integrate the detection function (now as a CUDA-compatible function).
2. **Set Up Hardware Configuration**:
   - In Simulink, go to **Model Configuration Parameters**.
   - Select **Hardware Implementation** and choose **NVIDIA Jetson** as the target hardware.
3. **Configure Camera and Sensor Inputs**: Ensure that the camera block in Simulink is compatible with the camera connected to the NVIDIA Jetson. You may need to use a **V4L2 Video Capture** block to access the camera input.

4. **Build and Deploy**:
   - Click **Deploy to Hardware** in Simulink. The model will be compiled and transferred to the Jetson board.
   - The detection algorithm will now run on the Jetson, accessing real-time camera data from the car's mounted camera.

### **5. Running the Model During Track Testing**

Once the model is deployed, it will begin running on the NVIDIA Jetson in real-time. During track testing:
- **The camera on the Formula Student car** will capture images of the track environment.
- The **deep learning detection model** running on the Jetson will process these images to identify cones and other relevant track features.
- **Output**: The detection results can be displayed on a screen connected to the car (if applicable) or saved to onboard storage for later analysis.

---

## **Conclusion**

This project provides a practical simulation setup for object detection using the Vehicle Dynamics Blockset™ in Simulink, with a focus on skidpad testing. By following these steps, you can verify that your deep learning algorithm accurately detects cones and assess its performance in terms of precision and efficiency.

---

## Refrences

1. **MathWorks Documentation on Vehicle Dynamics Blockset and Automated Driving Toolbox**  
   - MathWorks provides documentation for its tools, including the **Vehicle Dynamics Blockset** and **Automated Driving Toolbox**, which is useful for modeling vehicle dynamics and adding sensor simulations.
   - Vehicle Dynamics Blockset: [https://www.mathworks.com/products/vehicle-dynamics.html](https://www.mathworks.com/products/vehicle-dynamics.html)
   - Automated Driving Toolbox: [https://www.mathworks.com/products/automated-driving.html](https://www.mathworks.com/products/automated-driving.html)

2. **GPU Coder Documentation and NVIDIA Jetson Support**  
   - GPU Coder enables deployment of deep learning models to embedded devices like the NVIDIA Jetson. The documentation provides details on generating optimized CUDA code and deploying to NVIDIA GPUs.
   - GPU Coder: [https://www.mathworks.com/products/gpu-coder.html](https://www.mathworks.com/products/gpu-coder.html)
   - GPU Coder Support Package for NVIDIA GPUs: [https://www.mathworks.com/hardware-support/nvidia-gpu.html](https://www.mathworks.com/hardware-support/nvidia-gpu.html)

3. **Example on Deploying Deep Learning to NVIDIA Jetson and Other Hardware**  
   - MathWorks offers examples and tutorials on how to deploy deep learning models using GPU Coder. These examples demonstrate using tools like YOLO to NVIDIA hardware for real-time applications.
   - Example on Deploying Deep Learning on NVIDIA Jetson: [https://www.mathworks.com/help/gpucoder/examples/deploy-deep-learning-on-nvidia-gpu-hardware.html](https://www.mathworks.com/help/gpucoder/examples/deploy-deep-learning-on-nvidia-gpu-hardware.html)

4. **MathWorks Video Tutorials on Vehicle Dynamics Blockset and Deep Learning with Simulink**  
   - For a visual overview, MathWorks has videos that introduce the **Vehicle Dynamics Blockset** and demonstrate how to integrate deep learning models into Simulink for real-time applications.
   - Intro to Vehicle Dynamics Blockset: [https://www.mathworks.com/videos/what-is-vehicle-dynamics-blockset-1626005589938.html](https://www.mathworks.com/videos/what-is-vehicle-dynamics-blockset-1626005589938.html)

5. **General Information on YOLO (You Only Look Once) Object Detection**  
   - YOLO is a popular object detection framework that’s often used in automotive and real-time applications. The original YOLO papers and guides provide insights into how this algorithm works.
   - YOLO Paper: [https://arxiv.org/abs/1506.02640](https://arxiv.org/abs/1506.02640)

6. **NVIDIA Jetson Hardware Resources and Documentation**  
   - NVIDIA provides resources specifically for deploying AI and deep learning models on the Jetson platform. This includes information on supported frameworks, libraries, and CUDA optimization for embedded GPUs.
   - NVIDIA Jetson Developer Page: [https://developer.nvidia.com/embedded/jetson-developer](https://developer.nvidia.com/embedded/jetson-developer)

7. **vivinvarshan's Repository**  
   - Provides resources specifically for how to find neccessary add on and how to deploy them.
   - Page: [https://developer.nvidia.com/embedded/jetson-developer](https://github.com/vivinvarshans/Cone-Detection-MathWork-Assignment/tree/main)

I would like to show my gradtitude to VIVIN, who personally helped me on call with the neccessary add on and Skidpad test.

Each of these resources can provide additional details and practical guidance for implementing, deploying, and optimizing deep learning-based object detection systems in MATLAB/Simulink for real-time automotive applications.

