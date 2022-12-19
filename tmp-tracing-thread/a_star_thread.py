"""
20221216

Backend a-star algorithm.
This code was taken from V Jha's original implementation.
Here, the A* algorithm is extended to a Thread

Requires:
    numpy
    tifffile
    matplotlib

Running:

    ```
    cd tmp-tracing-thread
    python a_star_thread.py
    ```

    This will:
     - open a small tif file (from xxx)
     - Plot it in matplotlib
     - run the tracing thread
     - monitor the thread for output and update the plot

Notes:
    - The A* search algorithm takes some time to run and is blocking.
    We need to make it a thread `class aStarThread(Thread)`.
    We can communicate between threads with queue.Queue
        see self.outQueue

    - Added thread name to logger with `(%(threadName)-9s)`.
    Catching exceptions in Threads can be tricky, making debuggin hard.
    This does not solve that problem.

    - Once in PyQt/Napari, we can use a QTimer
    to monitor the A* thread and update the GUI interface at
    a given interval

    - We do want to implement bidirectional search. See:
    references/wink_et_al_2000.pdf

    - We want to optionally pre-process images with Hessian like filters
    See scikit ridge filter examples
    https://scikit-image.org/docs/stable/auto_examples/edges/plot_ridge_filter.html
    
"""

from collections import defaultdict
import math
from queue import PriorityQueue, Queue, Empty
import sys
import time
from threading import Thread
from typing import Dict, List, Tuple

import numpy as np
import tifffile

import matplotlib.pyplot as plt

import logging
logging.getLogger('matplotlib.font_manager').disabled = True
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s (%(threadName)-9s) %(filename)s:%(lineno)s - %(funcName)20s() %(message)s',)
logger = logging.getLogger()

class aStarThread(Thread):
    def __init__(self, imgData : np.ndarray,
                    startPnt : tuple, endPnt : tuple,
                    outQueue = None):
        super().__init__(daemon=True)

        self.imgData = imgData
        self.startPnt = startPnt
        self.endPnt = endPnt
        self.outQueue = outQueue

        self.step_size = 1
        
        # if 1 then we search a grid (no good)
        # if 2000, reasonable but get lots of bright pixel search away from endPnt
        # if 4000, very good, >3x faster than 2000 (not searching bright pixels away from target)
        # if 6000, about the same as 4000
        self.intScale = 4000  # scale int of neighbor for h score (use 4000)

        self.yrange = None
        self.xrange = None

        self.came_from = None
        self.current = None

        self.canceled = False

        self.timeout = 1  # seconds

    def cancel(self):
        """Cancel the thread worker, see run().

        TODO:
            When running in a GUI event loop,
            implement a cancel button widget.
        """
        self.canceled = True

    def set_ranges(self) -> None:
        """
        setter function for self.xrange and self.yrange
        """
        self.yrange= [0, self.imgData.shape[0]-1]
        self.xrange= [0, self.imgData.shape[1]-1]

    def get_neighbors(self, point: Tuple) -> List[Tuple]:
        """
        get 8 connected neighbors.
        """
        y, x = point
        y_min, y_max = self.yrange
        x_min, x_max = self.xrange
        neighbors = []

        step_size = self.step_size

        # top
        if y > y_min:
            neighbors.append((y - step_size, x))

        # bottom
        # if y < y_max - ERROR:
        if y < y_max:
            neighbors.append((y + step_size, x))

        # left
        if x > x_min:
            neighbors.append((y, x - step_size))

        # right
        # if x < x_max - ERROR:
        if x < x_max:
            neighbors.append((y, x + step_size))

        # top left
        if y > y_min and x > x_min:
            neighbors.append((y - step_size, x - step_size))
        # top right
        if y > y_min and x < x_max:
            neighbors.append((y - step_size, x + step_size))
        # bottom left
        if y < y_max and x > x_min:
            neighbors.append((y + step_size, x - step_size))
        # bottom right
        if y < y_max and x < x_max:
            neighbors.append((y + step_size, x + step_size))

        return neighbors

    def default_value(self) -> float:
        return float("inf")

    def calculate_h_score(self, p1: Tuple, p2: Tuple) -> float:
        """
        Returns the euclidean distance between two points
        scaled by the image intensity of p1
        """
        x1, y1 = p1
        x2, y2 = p2
        
        # euclidean
        dist =  math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # manhattan
        #dist = abs(x1 - x2) + abs(y1 - y2)

        #intScale = 2000  #500 # 1 causes a grid to be searched
        p1_int = self.imgData[p1[0], p1[1]] / self.intScale
        intensityCost = 1 / (p1_int)

        hScore = dist * intensityCost

        return hScore

    def is_close_to_end(self, point: Tuple,
                    error_threshold: float = 1.0) -> bool:
        dist = abs(point[0] - self.endPnt[0]) + abs(point[1] - self.endPnt[1])
        return dist <= error_threshold

    def getShortestPath_plot(self):  # -> List(int), List(int)
        """Get the shortest path after tracing finished.
        
        Returns:
            xPath: []
            yPath: []
        """
        if self.came_from is None:
            return None, None
        xPlotPath = []
        yPlotPath = []
        _current = self.current
        while _current in self.came_from:
            # self.add_point(np.array(current), PURPLE)
            #_pntList.append(current)
            xPlotPath.append(_current[1])
            yPlotPath.append(_current[0])
            _current = self.came_from[_current]
        return xPlotPath, yPlotPath

    def finishedTracing(self):
        return self.came_from is not None

    def getPathSearch_plot(self):
        """Get entire search.
        """
        if self.came_from is None:
            return None, None
        xPath = []
        yPath = []
        for k,v in self.came_from.items():
            xPath.append(v[1])
            yPath.append(v[0])
        return xPath, yPath

    def run(self):
        """
        run A* tracing algorithm
        """
        startPnt = self.startPnt
        endPnt = self.endPnt
        
        logger.info(f'Starting A* trace start:{startPnt} end:{endPnt}')

        startSec = time.time()

        step_size = self.step_size
        
        self.current = None
        self.came_from = None

        self.set_ranges()
        count = 0
        open_set = PriorityQueue()
        open_set.put(
            (0, count, startPnt)
        )  # distance, time of occurrence, point tuple
        came_from = {}
        g_score = defaultdict(self.default_value)
        g_score[startPnt] = 0
        f_score = defaultdict(self.default_value)
        f_score[startPnt] = self.calculate_h_score(startPnt, endPnt)

        open_set_hash = {startPnt}

        while not open_set.empty():
            if self.canceled:
                logger.info('canceled')
                break
            if (time.time() - startSec) > self.timeout:
                # TODO (cudmore) this does not work?
                # does time.time() work in a Thread?
                logger.info('timeout')
                break
            element = open_set.get()
            current = element[2]
            open_set_hash.remove(current)

            # if current == self.end:
            if self.is_close_to_end(current):
                stopSec = time.time()
                elapsedSec = round(stopSec-startSec,2)
                logger.info(f'found end point: {current} in {elapsedSec} sec.')
                self.current = current
                self.came_from = came_from
                return True

            # get 8-connected neighbors
            neighbors = self.get_neighbors(current)

            for neighbor in neighbors:
                temp_g_score = g_score[current] + step_size

                if temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + self.calculate_h_score(
                        neighbor, endPnt
                    )
                    if neighbor not in open_set_hash:
                        count += 1
                        if self.outQueue is not None:
                            # add to our out queue,
                            # can be monitored from caller
                            # to update plots
                            self.outQueue.put(neighbor)
                        open_set.put((f_score[neighbor], count, neighbor))
                        open_set_hash.add(neighbor)

        logger.info('Done tracing ... did not find path')
        return False

def _plotMplImage(tifData, start, end):
    plt.imshow(tifData, cmap='gray')
    plt.plot(start[1], start[0],'og')
    plt.plot(end[1], end[0], 'or')
    plt.pause(0.001)

def _updateMplyPlot(plotItems):
    """Update the points searched (cyan).

    Args:
        plotItems: [(y,x)]
    """
    yPlot = [x[0] for x in plotItems]
    xPlot = [x[1] for x in plotItems]

    plt.scatter(xPlot, yPlot, c='c', s=8, alpha=0.3)  # (x,y)
    plt.pause(0.001)  # run the plt event loop to get update

def _updateNapariSearch(searchLayer, plotItems : List):
    plotItems = np.array(plotItems)
    oldData = searchLayer.data  # fetch points from napari viewer
    if oldData.shape[0] == 0:
        newData = plotItems
    else:
        newData = np.concatenate((oldData, plotItems))
    searchLayer.data = newData

def _napari(imgData, startPnt, endPnt):
    import napari
    viewer = napari.Viewer()
    viewer.add_image(imgData, colormap='gray')

    startEndPoints = [
        [startPnt[0], startPnt[1]],
        [endPnt[0], endPnt[1]],
    ]
    _pointsStartEnd = viewer.add_points(startEndPoints, name='start end',
                                size=4, edge_width=1,
                                face_color="yellow", edge_color="yellow")

    _pointsSearched = viewer.add_points(None, name='searched',
                                size=2, edge_width=1,
                                face_color="green", edge_color="green")
    _pointsBrightestPath = viewer.add_points(None, name='brightest path',
                                size=2, edge_width=1,
                                face_color="cyan", edge_color="cyan")
    #napari.run()
    
    return viewer, _pointsSearched, _pointsBrightestPath

def _testAStarThread():
    doMatplotlib = True
    doNapari = False

    # small example
    # tifPath = 'testData/a-star-image-2.tif'
    # imgData = tifffile.imread(tifPath)
    # start = (128, 23)  # (y,x)
    # end = (23, 133)

    # medium example
    tifPath = 'testData/a-star-image.tif'
    imgData = tifffile.imread(tifPath)
    start = (188, 71)  # (y,x)
    end = (126, 701)

    if doMatplotlib:
        _plotMplImage(imgData, start, end)  # matplotlib
    if doNapari:
        _viewer, _pointsSearched, _pointsBrightestPath = _napari(imgData, start, end)

    outQueue = Queue()
    ast = aStarThread(imgData, start, end, outQueue=outQueue)
    ast.start()  # start the thread, internally Python calls tt.run()

    # control immediately returns here as A* runs in a different thread
    logger.info(f'1 TracingThread returned control to caller')
    
    # monitor outQueue for results
    #time.sleep(0.01)  # so we get something in the queue
    itemCount = 0
    _updateInterval = 500  # wait for this number of results and update plot
    plotItems = []
    while ast.is_alive() or not outQueue.empty():
        # when running in a GUI we can have a 'cancel' button
        #tt.cancel()
        
        try:
            item = outQueue.get(False)
            itemCount += 1
            # update a matplotlib/pyqtgraph/napari interface
            plotItems.append(item)
            if len(plotItems)>_updateInterval:
                if doMatplotlib:
                    _updateMplyPlot(plotItems)  # matplotlib
                if doNapari:
                    _updateNapariSearch(_pointsSearched, plotItems)
                plotItems = []

            # get the answer (shortest path)
            # if tt.finishedTracing():
            #     xPlotPath, yPlotPath = tt.getShortestPath_plot()
            #     plt.scatter(xPlotPath, yPlotPath, c='y', s=2)
            #     plt.pause(0.001)
        except Empty:
            # Handle empty queue here
            pass

    logger.info(f'2 TracingThread finished with {itemCount} items')
    
    # get the answer (shortest path) and plot it
    if ast.finishedTracing():
        xPlotPath, yPlotPath = ast.getShortestPath_plot()
        if doMatplotlib:
            plt.scatter(xPlotPath, yPlotPath, c='y', s=4)
            plt.pause(0.001)
        if doNapari:
            _pointsBrightestPath.data = np.column_stack((yPlotPath, xPlotPath))
    
    # keep the plot up
    if doMatplotlib:
        plt.show()

    if doNapari:
        napari.run()

if __name__ == '__main__':
    _testAStarThread()
    

