import h5py
import numpy as np
import tensorflow as tf

def load_datasets(file1, file2):
    train_dataset = h5py.File(file1, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:500]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:500]) # your train set labels

    val_set_x_orig = np.array(train_dataset["train_set_x"][500:]) # your train set features
    val_set_y_orig = np.array(train_dataset["train_set_y"][500:]) # your train set labels

    test_dataset = h5py.File(file2, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    val_set_y_orig = val_set_y_orig.reshape((1, val_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, val_set_x_orig, val_set_y_orig, test_set_x_orig, test_set_y_orig, classes
    

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True, directory='./', model_pb_name= 'my_model.pb'):
    """
    Freezes the state of a session into a pruned computation graph.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
    tf.train.write_graph(frozen_graph, directory, model_pb_name, as_text=False)
    print ('model written at '+directory+'/'+model_pb_name)