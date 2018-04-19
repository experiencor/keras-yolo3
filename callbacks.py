from keras.callbacks import Callback
import signal

class SignalStopping(Callback):
    '''Stop training when an interrupt signal (or other) was received
    # Arguments
        sig: the signal to listen to. Defaults to signal.SIGINT.
        doubleSignalExits: Receiving the signal twice exits the python
            process instead of waiting for this epoch to finish.
        verbose: verbosity mode.
    '''
    def __init__(self, sig=signal.SIGINT, doubleSignalExits=False, verbose=0):

        super(SignalStopping, self).__init__()

        self.signal_received = False
        self.verbose = verbose
        self.doubleSignalExits = doubleSignalExits

        def signal_handler(sig, frame):
            if self.signal_received and self.doubleSignalExits:
                if self.verbose > 0:
                    print('') #new line to not print on current status bar. Better solution?
                    print('Received signal to stop ' + str(sig) + ' twice. Terminating the training process...')
                exit(sig)

            self.signal_received = True
            self.model.stop_training = True
            if self.verbose > 0:
                print('') #new line to not print on current status bar. Better solution?
                print('Received signal to stop ' + str(sig) + '. Exiting fit_generator...')

        signal.signal(signal.SIGINT, signal_handler)