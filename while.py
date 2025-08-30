def get_grade(score):
    if score >= 90:
        return 'A+ (Excellent)'
    elif score >= 80:
        return 'A (Very Good)'
    elif score >= 70:
        return 'B (Good)'
    elif score >= 60:
        return 'C (Average)'
    elif score >= 50:
        return 'D (Pass)'
    else:
        return 'F (Fail)'

num_subjects = int(input("Enter total number of subjects: "))

# Store subjects and their marks in a dictionary
subject_marks = {}

for i in range(1, num_subjects + 1):
    # Input subject name
    while True:
        subject = input(f"Name of subject {i}: ").strip()
        if subject.replace(' ', '').isalpha():
            break
        print("Invalid input! Only letters allowed.")

    # Input marks with validation
    while True:
        try:
            marks = float(input(f"Enter marks obtained in {subject} (0-100): "))
            if 0 <= marks <= 100:
                break
            else:
                print("Marks must be between 0 and 100.")
        except ValueError:
            print("Enter a valid number for marks.")

    subject_marks[subject] = marks

# Calculate results
results = {}
for subject, marks in subject_marks.items():
    results[subject] = get_grade(marks)

# Calculate average and overall grade
average = sum(subject_marks.values()) / len(subject_marks)
overall = get_grade(average)
all_passed = all(m >= 50 for m in subject_marks.values())

# Display the report
print("\n--- Student Report ---")
for subject, grade in results.items():
    print(f"{subject:<15}: {subject_marks[subject]:>6.2f} marks | Grade: {grade}")

print(f"\nAverage Marks  : {average:.2f}")
print(f"Overall Grade  : {overall}")
print(f"Passed All     : {'Yes' if all_passed else 'No'}")
