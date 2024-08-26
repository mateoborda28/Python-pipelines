from iris_system import ML_logistic_Regression
import unittest

class test_iris(unittest.TestCase):
    def test_logistico(self):
        logistic = ML_logistic_Regression()
        resultado = logistic.ML_system_regression()
        self.assertTrue(resultado["success"], print("Modelo ejecutado correctamente"))
        self.assertGreaterEqual(resultado["accuracy"], 0.7, print("The model accuracy should be above 0.7"))
        print(resultado)
if __name__ == "__main__":
    unittest.main()
