from algorithm import AStarSearch
import time
from skimage import data
import numpy as np
import napari
from thread import AStarThread

def test_2D_image():
    # testing for 2D
    #twoDImage = data.cells3d()[30, 1]  # brighter image
    twoDImage = data.cells3d()[30, 0] # darker image
    # twoDImage = twoDImage / np.max(twoDImage) * 255
    # twoDImage = twoDImage.astype(np.uint8)
    
    # start_point = np.array([13,0]) # y, x in 2d
    # goal_point = np.array([45,90]) # y, x
    twoDImage = twoDImage[:5, :5]
    start_point = np.array([0,0])
    goal_point = np.array([4,4])
    print(f"image: {twoDImage}")

    astar_search = AStarSearch(
        image=twoDImage,
        start_point=start_point,
        goal_point=goal_point)
    tic = time.perf_counter()
    result = astar_search.search()
    toc = time.perf_counter()
    print(f"Found brightest path in {toc - tic:0.4f} seconds")
    print(f"path size: {len(result)}")
    print(f"path: {result}")

    # viewer = napari.Viewer()
    # viewer.add_image(twoDImage[:100, :250], colormap='magma')
    # viewer.add_points(np.array([start_point, goal_point]), size=2, edge_width=1, face_color="white", edge_color="white")
    # viewer.add_points(result, size=1, edge_width=1, face_color="green", edge_color="green")
    # napari.run()

def test_2D_image_thread():
    # testing for 2D using thread
    twoDImage = data.cells3d()[30, 0] # darker image
    # twoDImage = twoDImage[:5, :5]
    
    start_point = np.array([13,0]) # y, x in 2d
    goal_point = np.array([45,90]) # y, x

    astar_thread = AStarThread(
        image=twoDImage,
        start_point=start_point,
        goal_point=goal_point)

    tic = time.perf_counter()
    astar_thread.run()
    astar_thread.join()
    toc = time.perf_counter()
    print(f"Found brightest path in {toc - tic:0.4f} seconds")
    print(f"path size: {len(astar_thread.result)}")
    print(f"path: {astar_thread.result}")

    # viewer = napari.Viewer()
    # viewer.add_image(twoDImage[:100, :250], colormap='magma')
    # viewer.add_points(np.array([start_point, goal_point]), size=2, edge_width=1, face_color="white", edge_color="white")
    # viewer.add_points(result, size=1, edge_width=1, face_color="green", edge_color="green")
    # napari.run()

def test_3D_image():
    # testing for 3D
    threeDImage = data.cells3d()[30]  # grab some data
    # start_point = np.array([0, 13, 0]) # z, x, y in 3D
    # goal_point = np.array([0, 45, 90])
    threeDImage = threeDImage[:2, :5, :5]
    start_point = np.array([0,0,0])
    goal_point = np.array([0,4,4])
    print(f"image: {threeDImage}")

    astar_search = AStarSearch(
        image=threeDImage,
        start_point=start_point, 
        goal_point=goal_point,
        )

    result = astar_search.search()
    print(f"path size: {len(result)}")
    print(f"path: {result}")

    # viewer = napari.Viewer()
    # viewer.add_image(threeDImage[:100, :250], colormap='magma')
    # viewer.add_points(np.array([start_point, goal_point]), size=2, edge_width=1, face_color="white", edge_color="white")
    # viewer.add_points(result, size=1, edge_width=1, face_color="green", edge_color="green")
    # napari.run()

if __name__ == "__main__":
    # test_2D_image()
    # test_3D_image()
    test_2D_image_thread()
