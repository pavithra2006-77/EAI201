class SmartVacuum:
    def __init__(self, shape):
        self.shape = shape
        self.state = "idle"
        self.battery = 100  # battery percentage

    def command(self, action):
        if action == "start":
            if self.battery <= 0:
                print("Battery empty! Please dock to recharge.")
                return
            self.state = "cleaning"
            print(f"{self.shape} vacuum started cleaning.")
            self.clean_mode()
            self.battery -= 10
        elif action == "stop":
            self.state = "idle"
            print(f"{self.shape} vacuum stopped.")
        elif action == "left":
            print(f"{self.shape} vacuum turned left.")
        elif action == "right":
            print(f"{self.shape} vacuum turned right.")
        elif action == "dock":
            self.state = "docking"
            self.battery = 100
            print(f"{self.shape} vacuum docking at charging station. Battery recharged to 100%.")
        elif action == "status":
            self.display_status()
        else:
            print(f"Action '{action}' not recognized.")

    def display_status(self):
        print(f"Vacuum State: {self.state}, Shape: {self.shape}, Battery: {self.battery}%")

    def clean_mode(self):
        print("\nCleaning categories: 1. Solid  2. Liquid")
        try:
            choice = int(input("Enter choice number: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            return

        if choice == 1:
            self.solid_clean()
        elif choice == 2:
            self.liquid_clean()
        else:
            print("Invalid category.")

    def solid_clean(self):
        print("\nSolid options: 1. Dust  2. Debris  3. Mixed")
        try:
            opt = int(input("Enter solid type: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            return

        if opt == 1:
            self.shape = "Oval"
            print("Cleaning dust... Shape optimized to Oval for efficient coverage.")
        elif opt == 2:
            self.shape = "Circle"
            print("Cleaning debris... Shape optimized to Circle for stronger suction.")
        elif opt == 3:
            self.shape = "Hexagon"
            print("Cleaning mixed solids... Shape changed to Hexagon for stability.")
        else:
            print("Invalid solid option.")

    def liquid_clean(self):
        print("\nLiquid options: 1. Water  2. Oil/Other")
        try:
            opt = int(input("Enter liquid type: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            return

        if opt == 1:
            self.shape = "Funnel"
            print("Cleaning water... Shape changed to Funnel for flow efficiency.")
        elif opt == 2:
            self.shape = "Cone"
            print("Cleaning other liquids... Shape changed to Cone for better suction.")
        else:
            print("Invalid liquid option.")


# --- Main Program ---
available_shapes = ["Circle", "Square", "Triangle", "Pentagon"]
print("Available vacuum shapes:", available_shapes)

user_shape = input("Choose your vacuum shape: ").capitalize()

if user_shape in available_shapes:
    vac = SmartVacuum(user_shape)

    while True:
        print("\nActions: start | left | right | dock | stop | status | exit")
        action = input("Enter action: ").lower()
        if action == "exit":
            print("Exiting program.")
            break
        vac.command(action)

        if action == "stop":
            print("Session ended.")
            break
else:
    print("Invalid shape. Please restart and choose from the given list.")
