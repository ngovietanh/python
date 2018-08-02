import tensorflow as tf 
import numpy as np 
import argparse
import loaddata

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def my_model(features, labels, mode, params):    
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)  
    ## activation function
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    ## loss function
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
     
    ## result measurement
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
    # args = parser.parse_args(argv[1:])    
    # data = loaddata.read_data()    
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess :
        sess.run(init)
        print("Starttraining")
    
    my_feature_columns = []
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,            
            'hidden_units': [10, 10],            
            'n_classes': 3,
        }) 
    
    # classifier.train(
    #     input_fn=lambda:iris_data.train_input_fn(train_x, train_y, args.batch_size),
    #     steps=args.train_steps)
    # return 0
    
    # eval_result = classifier.evaluate(
    #     input_fn=lambda:iris_data.eval_input_fn(test_x, test_y, args.batch_size))

    # print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)


