import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow as  tf2
def rename_var(ckpt_path, new_ckpt_path):
    with tf.Session() as sess:
        for var_name, _ in tf2.train.list_variables(ckpt_path):
            print(var_name)
            var = tf2.train.load_variable(ckpt_path, var_name)
            new_var_name = var_name.replace('/', '_')
            var = tf.Variable(var, name=new_var_name)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, new_ckpt_path)
if __name__ == '__main__':
    ckpt_path=[]
    new_ckpt_path=[]
    ckpt_path.append    ('/home/zjj/WTUDF/weight_tf/wgts_epochs_10000.ckpt')
    new_ckpt_path.append('/home/zjj/WTUDF/weight_tf/wgts_epochs_10000_fix.ckpt')
    for ckpt_1,ckpt_fix in zip(ckpt_path,new_ckpt_path):
        rename_var(ckpt_1, ckpt_fix)

