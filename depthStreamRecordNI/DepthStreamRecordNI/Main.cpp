#include "DepthStreamRecord.h"
#include "NIViewer.h"

int main(int argc, char** argv)
{
	Status status = STATUS_OK;
	printf("Scanning machine for devices and loading "
		"modules/drivers ...\r\n");

	status = OpenNI::initialize();
	if (!HandleStatus(status)) return 1;
	printf("Completed.\r\n");

	Device device;
	printf("Opening first device ...\r\n");
	status = device.open(ANY_DEVICE);
	if (!HandleStatus(status)) return 1;
	printf("%s Opened, Completed.\r\n",
		device.getDeviceInfo().getName());

	printf("Checking if stream is supported ...\r\n");
	if (!device.hasSensor(SENSOR_DEPTH))
	{
		printf("Stream not supported by this device.\r\n");
		return 1;
	}

	printf("Asking device to create a depth stream ...\r\n");
	VideoStream depthSensor;
	status = depthSensor.create(device, SENSOR_DEPTH);
	if (!HandleStatus(status)) return 1;

	printf("Asking device to create a color stream ...\r\n");
	VideoStream colorSensor;
	status = colorSensor.create(device, SENSOR_COLOR);
	if (!HandleStatus(status)) return 1;

	if (device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR)!= openni::STATUS_OK) return 1;

	printf("Starting depth stream ...\r\n");
	status = depthSensor.start();
	if (!HandleStatus(status)) return 1;
	printf("Done.\r\n");

	printf("Starting color stream ...\r\n");
	status = colorSensor.start();
	if (!HandleStatus(status)) return 1;
	printf("Done.\r\n");

	printf("Creating a depth recorder ...\r\n");
	Recorder recorder;
	status = recorder.create("record.oni");
	if (!HandleStatus(status)) return 1;
	printf("Done.\r\n");
	printf("Attaching to depth sensor ...\r\n");
	status = recorder.attach(depthSensor);
	if (!HandleStatus(status)) return 1;
	status = recorder.attach(colorSensor);
	if (!HandleStatus(status)) return 1;
	printf("Done.\r\n");

	//printf("Creating a color recorder ...\r\n");
	//Recorder colorRecorder;
	//status = colorRecorder.create("color.oni");
	//if (!HandleStatus(status)) return 1;
	//printf("Done.\r\n");
	//printf("Attaching to depth sensor ...\r\n");
	//status = colorRecorder.attach(colorSensor);
	//if (!HandleStatus(status)) return 1;
	//printf("Done.\r\n");

	printf("Starting recorder ...\r\n");
	status = recorder.start();
	if (!HandleStatus(status)) return 1;
	printf("Done. Now recording ...\r\n");

	NIViewer matContainer("MATs Container", device, depthSensor, colorSensor);
	status = matContainer.init();
	if (!HandleStatus(status)) return 1;
	
	//Setting up auto white balance and auto exposure
	{
		CameraSettings camSetting = *(colorSensor.getCameraSettings());

		camSetting.setAutoExposureEnabled(false);
		camSetting.setAutoWhiteBalanceEnabled(false);
	}

	while (1)
	{
		matContainer.run();

		cv::Mat colorMat = matContainer.getColorMat();
		cv::Mat depthMat = matContainer.getDepthMat8();
		
		cv::imshow("color", colorMat);
		cv::imshow("depth", depthMat);

		if (cvWaitKey(30) >= 0)	break;
	}

	recorder.destroy();
	depthSensor.destroy();
	colorSensor.destroy();
	device.close();
	OpenNI::shutdown();
}