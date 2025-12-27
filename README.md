Project Structure :
layers.py

    1- Affine

    2- BatchNormalization

    3- Dropout

activationsLosses.py

    1- ReLU

    2- Sigmoid

    3- SoftmaxWithLoss

    4- MeanSquaredError

optimizers.py

    1- SGD

    2- Momentum

    3- AdaGrad

    4- Adam

neural_network.py

    1- NeuralNetwork class

HyperparameterTuning.py

    1- HyperparameterTuning class

trainer.py

    1- train_step

    2- fit

main.py

ملف الاختبار الأساسي (MNIST)



🚀 كيفية التشغيل (How to Run)
تأكد من تثبيت المكتبات اللازمة:


pip install numpy matplotlib sklearn

تشغيل الاختبار الأساسي: قم بتشغيل ملف main.py لبدء عملية البحث عن البارامترات العليا ثم تدريب الشبكة المطلوبة:


python main.py