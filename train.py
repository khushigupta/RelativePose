import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model, save_model
from siamese_model import match_model, pose_model, identity_loss
import argparse
import datetime


def get_model(model_type, **kwargs):

    if model_type == 'match':
        input_shape = (None, 64, 64, 3)
        label_shape = (None, 1)
        match = match_model(input_shape, label_shape)
        match.compile(optimizer='sgd',
                      loss=identity_loss,
                      metrics=['accuracy'])


    if model_type == 'pose':

        input_shape = (None, 224, 224, 3)

        if 'match_model' not in kwargs:
            pose = pose_model(input_shape)
        else:
            match_pretrained = kwargs['match_model']
            pose = pose_model(input_shape, match_pretrained)

        pose.compile(optimizer='sgd',
                     loss=['mean_squared_error', 'mean_squared_error'],
                     metrics=['accuracy'],
                     loss_weights=[1., 1])

        return pose


def train(model, df_train, train_dir, fit_index, val_index, label_map,
                model_weights_path, data_gen_args_fit={}, data_gen_args_val={}, seed=2017):


    callbacks = [EarlyStopping(monitor='val_loss', patience=4, verbose=1),
                 ModelCheckpoint(model_weights_path, monitor='val_loss',
                                 save_best_only=True, verbose=0)]

    steps_per_epoch_fit = np.ceil(len(fit_index) / args.batch_size)
    steps_per_epoch_val = np.ceil(len(val_index) / args.batch_size)

    fit_generator = batch_generator(train_dir,
                                    df_train.iloc[fit_index],
                                    label_map,
                                    batch_size=args.batch_size,
                                    number_of_batches=steps_per_epoch_fit,
                                    data_gen_args=data_gen_args_fit,
                                    seed=seed)

    val_generator = batch_generator(train_dir,
                                    df_train.iloc[val_index],
                                    label_map,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    number_of_batches=steps_per_epoch_val,
                                    data_gen_args=data_gen_args_val)
    try:
        model.fit_generator(generator=fit_generator,
                            steps_per_epoch=steps_per_epoch_fit,
                            epochs=args.epochs,
                            verbose=1,
                            validation_data=val_generator,
                            validation_steps=steps_per_epoch_val,
                            callbacks=callbacks)
    except KeyboardInterrupt:
        pass

    return model


def evaluate():
    pass


def predict():
    pass


def get_name():

    time_str = datetime.now().strftime("%M_%H_%d")
    model_name = '{}/{}_{}.h5'.format(args.path, args.model_type, time_str)
    return model_name


def main():

    if args.model_type == 'match':
        model = get_model('match')

    else:
        if args.pretrain_path is not None:
            match_trained = load_model(args.pretrain_path)
            model = get_model('match', pretrain=match_trained)
        else:
            model = get_model('match')

    if args.train:
        trained_model = train(model)
        save_model(get_name(), trained_model)

        print('Finished training!')

    if args.evaluate:
        model = load_model(args.e_model)
        evaluate(model)
        print('Finished Evaluating!')

    if args.predict:
        model = load_model(args.p_model)
        predict(model)
        print('Finished Predicting!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('--cuda', action='store_true', default=False, help='Enables CUDA training')
    parser.add_argument('--train', type=int, default=True, help='Train model')
    parser.add_argument('--model_type', type=str, default='', help='Train Pose/SIFT model')
    parser.add_argument('--pretrain_path', type=str, default='', help='Path to pretrained SIFT model')

    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')

    parser.add_argument('--path', type=str, default='', help='Path to store trained models')
    parser.add_argument('--log_path', type=str, default='', help='Path to store logs')

    parser.add_argument('--evaluate', type=int, default=False, help='Evaluate model')
    parser.add_argument('--e_model', type=int, default=False, help='Model to evaluate')

    parser.add_argument('--predict', type=int, default=False, help='Get predictions')
    parser.add_argument('--p_model', type=int, default=False, help='Model to use for predictions')

    args = parser.parse_args()

    main()


