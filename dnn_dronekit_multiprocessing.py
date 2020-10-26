# import the necessary packages
from keras.models import load_model
import argparse
import pickle
import cv2
import time
import math
from picamera.array import PiRGBArray
from picamera import PiCamera
from dronekit import connect, VehicleMode, LocationGlobalRelative, LocationGlobal
from pymavlink import mavutil
import signal
import multiprocessing

def to_quaternion(roll = 0.0, pitch = 0.0, yaw = 0.0):
    """
    Convert degrees to quaternions
    """
    t0 = math.cos(math.radians(yaw * 0.5))
    t1 = math.sin(math.radians(yaw * 0.5))
    t2 = math.cos(math.radians(roll * 0.5))
    t3 = math.sin(math.radians(roll * 0.5))
    t4 = math.cos(math.radians(pitch * 0.5))
    t5 = math.sin(math.radians(pitch * 0.5))

    w = t0 * t2 * t4 + t1 * t3 * t5
    x = t0 * t3 * t4 - t1 * t2 * t5
    y = t0 * t2 * t5 + t1 * t3 * t4
    z = t1 * t2 * t4 - t0 * t3 * t5

    return [w, x, y, z]

def cameraCaptureFunction(grayQueue, colorQueue):
    camera = PiCamera()
    camera.resolution = (640,480)
    camera.framerate = 15
    rawCapture = PiRGBArray(camera, size=(640, 480))

    # allow the camera to warmup
    time.sleep(0.1)

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

        image = frame.array
        image = cv2.transpose(image)
        image = cv2.flip(image, 1)
        image = image[200:-200, 160:-160]
        colorImage = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (100, 100))

        image = image.astype("float") / 255.0

        image = image.reshape((1, image.shape[0], image.shape[1], 1))

        try:
            grayQueue.put_nowait(image)
            colorQueue.put_nowait(colorImage)
        except:
            pass

        rawCapture.truncate(0)

        time.sleep(0.05)

        if mainExit.is_set():
            break
    print("camera exit")

def moveFunction(moveCommand):
    # dronekit comms================================================================================================

    # Set up option parsing to get connection string
    import argparse
    parser = argparse.ArgumentParser(description='Commands vehicle using vehicle.simple_goto.')
    parser.add_argument('--connect',
                        help="Vehicle connection target string. If not specified, SITL automatically started and used.")
    parser.add_argument('--baudrate',
                        help="baudrate")
    args = parser.parse_args()

    connection_string = args.connect
    connection_baud = args.baudrate
    sitl = None


    # Start SITL if no connection string specified
    if not connection_string:
        import dronekit_sitl
        sitl = dronekit_sitl.start_default()
        connection_string = sitl.connection_string()


    # Connect to the Vehicle
    print('Connecting to vehicle on: %s' % connection_string)
    vehicle = connect(connection_string, wait_ready=True, baud=connection_baud)

    def send_attitude_target(roll_angle = 0.0, pitch_angle = 0.0,
                             yaw_angle = None, yaw_rate = 0.0, use_yaw_rate = False,
                             thrust = 0.5):
        """
        use_yaw_rate: the yaw can be controlled using yaw_angle OR yaw_rate.
                      When one is used, the other is ignored by Ardupilot.
        thrust: 0 <= thrust <= 1, as a fraction of maximum vertical thrust.
                Note that as of Copter 3.5, thrust = 0.5 triggers a special case in
                the code for maintaining current altitude.
        """
        if yaw_angle is None:
            # this value may be unused by the vehicle, depending on use_yaw_rate
            yaw_angle = vehicle.attitude.yaw
        # Thrust >  0.5: Ascend
        # Thrust == 0.5: Hold the altitude
        # Thrust <  0.5: Descend
        msg = vehicle.message_factory.set_attitude_target_encode(
            0, # time_boot_ms
            1, # Target system
            1, # Target component
            0b00000000 if use_yaw_rate else 0b00000100,
            to_quaternion(roll_angle, pitch_angle, yaw_angle), # Quaternion
            0, # Body roll rate in radian
            0, # Body pitch rate in radian
            math.radians(yaw_rate), # Body yaw rate in radian/second
            thrust  # Thrust
        )
        vehicle.send_mavlink(msg)

    while not mainExit.is_set():
        command = moveCommand.get()
        if command == 1:
            send_attitude_target(roll_angle = 1, thrust = 0.5)
        elif command == 2:
            send_attitude_target(roll_angle = -1, thrust = 0.5)
        elif command == 3:
            send_attitude_target(thrust = 0.55)
        elif command == 4:
            send_attitude_target(thrust = 0.45)
        elif command == 5:
            send_attitude_target(pitch_angle = 1, thrust = 0.5)
        elif command == 6:
            send_attitude_target(pitch_angle = -1, thrust = 0.5)
        elif command == 7:
            pass
        time.sleep(0.2)

    print("Close vehicle object")
    vehicle.close()


if __name__ == "__main__":
    mainExit = multiprocessing.Event()

    grayFrames = multiprocessing.Queue(maxsize = 2)
    colorFrames = multiprocessing.Queue(maxsize = 2)
    inferenceResult = multiprocessing.Queue(maxsize = 2)

    cameraCapture = multiprocessing.Process(
        target = cameraCaptureFunction,
        kwargs = {
            'grayQueue': grayFrames,
            'colorQueue': colorFrames
        }
    )
    cameraCapture.start()

    moving = multiprocessing.Process(
        target = moveFunction,
        kwargs = {
            'moveCommand': inferenceResult
        }
    )
    moving.start()

    print("[INFO] loading network and label binarizer...")
     

    print("Starting Deep Neural Network loop. . .")

    frameCounter = 0
    fps = 0
    startTime = time.time()
    while not mainExit.is_set():
        frameCounter = frameCounter + 1

        image = grayFrames.get()
        output = colorFrames.get()

        # make a prediction on the image
        preds = model.predict(image)

        i = preds.argmax(axis=1)[0]
        label = lb.classes_[i]

        if preds[0][i] < 0.95:
            label = "idle"


        text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
        cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if frameCounter == 100:
            # every 10 frames, calculate the framerate
            fps = 1.0 / ((time.time() - startTime) / frameCounter)
            frameCounter = 0
            startTime = time.time()

        cv2.putText(output, "{:.2f}".format(fps), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Frame',output)

        try:
            if label == "left":
                inferenceResult.put_nowait(1)
            elif label == "right":
                inferenceResult.put_nowait(2)
            elif label == "up":
                inferenceResult.put_nowait(3)
            elif label == "down":
                inferenceResult.put_nowait(4)
            elif label == "forward":
                inferenceResult.put_nowait(5)
            elif label == "back":
                inferenceResult.put_nowait(6)
            elif label == "idle":
                inferenceResult.put_nowait(7)

        except:
            pass

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            mainExit.set()
            break

    print("Shutting Down")
