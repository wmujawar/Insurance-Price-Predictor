import pytest
from src.app import _validate_input

def test_valid_input():
    data = {
        'Age': 25,
        'Diabetes': 1,
        'BloodPressureProblems': 0,
        'AnyTransplants': 0,
        'AnyChronicDiseases': 1,
        'KnownAllergies': 0,
        'HistoryOfCancerInFamily': 1,
        'Height': 170,
        'Weight': 70,
        'NumberOfMajorSurgeries': 1
    }
    response, status_code, headers = _validate_input(data)
    assert response is None
    assert status_code is None
    assert headers is None

@pytest.mark.parametrize("age", [17, 111])
def test_invalid_age(age):
    data = {
        'Age': age,
        'Diabetes': 1,
        'BloodPressureProblems': 0,
        'AnyTransplants': 0,
        'AnyChronicDiseases': 1,
        'KnownAllergies': 0,
        'HistoryOfCancerInFamily': 1,
        'Height': 170,
        'Weight': 70,
        'NumberOfMajorSurgeries': 1
    }
    response, status_code, headers = _validate_input(data)
    assert response == "Age must be between 18 and 110."
    assert status_code == 400
    assert headers['Content-Type'] == 'text/plain'

@pytest.mark.parametrize("field, value, error_message", [
    ('Diabetes', 2, "Diabetes must be either 0 or 1."),
    ('BloodPressureProblems', 2, "BloodPressureProblems must be either 0 or 1."),
    ('AnyTransplants', 2, "AnyTransplants must be either 0 or 1."),
    ('AnyChronicDiseases', 2, "AnyChronicDiseases must be either 0 or 1."),
    ('KnownAllergies', 2, "KnownAllergies must be either 0 or 1."),
    ('HistoryOfCancerInFamily', 2, "HistoryOfCancerInFamily must be either 0 or 1.")
])
def test_invalid_binary_field(field, value, error_message):
    data = {
        'Age': 25,
        'Diabetes': 1,
        'BloodPressureProblems': 0,
        'AnyTransplants': 0,
        'AnyChronicDiseases': 1,
        'KnownAllergies': 0,
        'HistoryOfCancerInFamily': 1,
        'Height': 170,
        'Weight': 70,
        'NumberOfMajorSurgeries': 1
    }
    data[field] = value
    response, status_code, headers = _validate_input(data)
    assert response == error_message
    assert status_code == 400
    assert headers['Content-Type'] == 'text/plain'

@pytest.mark.parametrize("field, value, error_message", [
    ('Height', -1, "Height and Weight must be non-negative."),
    ('Weight', -1, "Height and Weight must be non-negative.")
])
def test_negative_height_weight(field, value, error_message):
    data = {
        'Age': 25,
        'Diabetes': 1,
        'BloodPressureProblems': 0,
        'AnyTransplants': 0,
        'AnyChronicDiseases': 1,
        'KnownAllergies': 0,
        'HistoryOfCancerInFamily': 1,
        'Height': 170,
        'Weight': 70,
        'NumberOfMajorSurgeries': 1
    }
    data[field] = value
    response, status_code, headers = _validate_input(data)
    assert response == error_message
    assert status_code == 400
    assert headers['Content-Type'] == 'text/plain'

def test_invalid_number_of_major_surgeries():
    data = {
        'Age': 25,
        'Diabetes': 1,
        'BloodPressureProblems': 0,
        'AnyTransplants': 0,
        'AnyChronicDiseases': 1,
        'KnownAllergies': 0,
        'HistoryOfCancerInFamily': 1,
        'Height': 170,
        'Weight': 70,
        'NumberOfMajorSurgeries': 4
    }
    response, status_code, headers = _validate_input(data)
    assert response == "NumberOfMajorSurgeries must be between 0 and 3."
    assert status_code == 400
    assert headers['Content-Type'] == 'text/plain'