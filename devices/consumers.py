import json
import os
from channels.generic.websocket import AsyncWebsocketConsumer
import base64
import cv2
import asyncio
import numpy as np
from channels.db import database_sync_to_async

from _smarthome.settings import BASE_DIR
from devices.models import Camera
from . import views
from imutils.video import VideoStream, WebcamVideoStream
from imutils.video import FPS
from collections import OrderedDict


class TrackableObject:
    def __init__(self, objectID, centroid):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]
        # initialize a boolean used to indicate if the object has
        # already been counted or not
        self.counted = False


class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = np.linalg.norm(
                np.array(objectCentroids)[:, np.newaxis] - inputCentroids, axis=2
            )

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for row, col in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects


class CameraStreamConsumer(AsyncWebsocketConsumer):
    def get_camera(self):
        return Camera.objects.aget(pk=self.scope["url_route"]["kwargs"]["pk"])

    async def connect(self):
        await self.accept()
        self.streaming = True
        self.camera = await database_sync_to_async(self.get_camera)()
        asyncio.create_task(self.send_video_stream())

    async def disconnect(self, close_code):
        self.streaming = False

    def tag_on():
        return 1

    async def send_video_stream(self):
        vc = cv2.VideoCapture(os.path.join(BASE_DIR, "mobilenet_ssd", "test.mp4"))
        # vc = cv2.VideoCapture(0)
        prototxt = os.path.join(
            BASE_DIR, "mobilenet_ssd", "MobileNetSSD_deploy.prototxt"
        )
        caffemodel = os.path.join(
            BASE_DIR, "mobilenet_ssd", "MobileNetSSD_deploy.caffemodel"
        )

        CLASSES = [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]

        net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

        W = None
        H = None

        ct = CentroidTracker()

        trackers = []
        trackableObjects = {}

        totalFrames = 0
        totalDown = 0
        totalUp = 0
        totalLeft = 0
        totalRight = 0

        fps = FPS().start()

        while self.streaming:
            ret, frame = vc.read()

            if ret is None:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            status = "Waiting"
            rects = []

            if totalFrames % 30 == 0:
                status = "Detecting"
                trackers = []

                blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
                net.setInput(blob)
                detections = net.forward()

                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        idx = int(detections[0, 0, i, 1])
                        if CLASSES[idx] != "person":
                            continue

                        box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                        (startX, startY, endX, endY) = box.astype("int")
                        cv2.rectangle(
                            frame, (startX, startY), (endX, endY), (0, 255, 0), 2
                        )
                        y = startY - 10 if startY - 10 > 10 else startY + 10

                        cv2.putText(
                            frame,
                            CLASSES[idx],
                            (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                        # tracker = dlib.correlation_tracker()
                        # rect = dlib.rectangle(startX, startY, endX, endY)
                        # tracker.start_track(rgb, rect)

                        # trackers.append(tracker)

            else:
                for tracker in trackers:
                    status = "Tracking"
                    tracker.update(rgb)
                    pos = tracker.get_position()

                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    rects.append((startX, startY, endX, endY))

            cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
            cv2.line(frame, (W // 2, 0), (W // 2, H), (255, 255, 255), 2)

            objects = ct.update(rects)
            for objectID, centroid in objects.items():
                to = trackableObjects.get(objectID, None)
                if to is None:
                    to = TrackableObject(objectID, centroid)
                else:
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)

                    if not to.counted:
                        cv2.putText(
                            frame,
                            "Left -> Right",
                            (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )
                        if direction < 0 and centroid[1] < H // 2:
                            totalUp += 1
                            to.counted = True

                        if direction > 0 and centroid[1] > H // 2:
                            totalDown += 1
                            to.counted = True

                        # Di chuyển trái sang phải
                        if (
                            direction < 0 and centroid[0] > W // 2
                        ):  # Di chuyển sang phải
                            totalRight += 1  # Tăng số lượng người di chuyển sang phải
                            to.counted = True  # Đánh dấu đối tượng là đã đếm

                        # Di chuyển phải sang trái
                        if (
                            direction > 0 and centroid[0] < W // 2
                        ):  # Di chuyển sang trái
                            totalLeft += 1  # Tăng số lượng người di chuyển sang trái
                            to.counted = True  # Đánh dấu đối tượng là đã đếm

                trackableObjects[objectID] = to

                text = "ID {}".format(objectID)
                cv2.putText(
                    frame,
                    text,
                    (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            info = [
                ("Up", totalUp),
                ("Down", totalDown),
                ("Left", totalLeft),
                ("Right", totalRight),
                ("Status", status),
            ]

            for i, (k, v) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(
                    frame,
                    text,
                    (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            totalFrames += 1
            fps.update()

            _, buffer = cv2.imencode(".jpg", frame)

            jpg_as_text = base64.b64encode(buffer).decode("utf-8")
            await self.send(text_data=json.dumps({"jpg_as_text": jpg_as_text}))
            await asyncio.sleep(0.01)

        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        vc.release()
