from dialogue_system import translate
from sentence_controller import evaluate

def main():
    while True:
        input_msg = input("input : ")
        print("out : {}".format(translate.generate_answer(input_msg)))
        
        p_sent, n_sent = evaluate.control(input_msg)
        print("pos : {}".format(p_sent))
        print("neg : {}".format(n_sent))

if __name__ == "__main__":
    main()
