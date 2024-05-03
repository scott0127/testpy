import Blackboard
def main():
    # Create a blackboard object
    blackboard = Blackboard.Blackboard("./image/picture.jpg")
    blackboard.run()
    
    blackboard.show()
    pass


if __name__ == "__main__":
    main()