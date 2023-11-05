import model_tester 

tester = model_tester.Tester(None,proc_mode="cuda")
tester.LoadModel()
while True:
    try:
        index = input("Please enter index number: ")
        print("q")
        if( index == 'q'):
            break
        index = int(index)

        if( index>=0 & index<8664 ):
            tester.TestIndex(index)
        elif( index ==-1 ):
            tester.TestCLI()

        continue
    except ValueError:
        print("Sorry, I didn't understand that.")
        continue

    