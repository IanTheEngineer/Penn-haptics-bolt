import tables
import numpy as np

def create_labeled_data(processor = None):
    filenames = ("bouncy_foam.h5", 
                 "cork.h5", "glass_bottle.h5", 
                 "hard_rough_foam.h5", 
                 "metal_bar.h5", 
                 "soft_foam.h5")
    
    labeled = {}
    for filename in filenames:
        all_data = tables.openFile(filename)
        trajectories = [ _g for _g in all_data.walkGroups("/") if _g._v_depth == 1]
        fingers_0 = [g.finger_0.electrodes.read() for g in trajectories]
        fingers_1 = [g.finger_1.electrodes.read() for g in trajectories]
        all_fingers = [ np.hstack((f0, f1))for (f0,f1) in zip(fingers_0, fingers_1)]
        
        name = filename.partition(".")[0]
        if processor is not None:
            data = [processor.transform(x) for x in all_fingers]
        else:
            data = all_fingers
        labeled[name] = data
    
    return labeled

def create_training_labels(labeled_data):
    isinstance(labeled_data, dict)
    
    points = []
    labels = []
    allkeys = labeled_data.keys() 
    for k in allkeys:
        for p in labeled_data[k]:
            points.append(p)
            labels.append(allkeys.index(k))
            
    return points, labels