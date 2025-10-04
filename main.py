from collect import collect_data
from train import train_model
from recognize import recognize_face

if __name__ == "__main__":
    while True:
        print("\n==========================")
        print(" HỆ THỐNG NHẬN DIỆN KHUÔN MẶT ")
        print("==========================")
        print("1. Thu thập dữ liệu mặt (vui lòng để mặt trước camera để lấy dữ liệu)")
        print("2. Train mô hình ")
        print("3. Nhận diện khuôn mặt")
        print("0. Thoát")
        choice = input("Chọn chức năng: ")

        if choice == "1":
            collect_data()
        elif choice == "2":
            train_model()
        elif choice == "3":
            recognize_face()
        elif choice == "0":
            print("Tạm biệt!")
            break
        else:
            print("Lựa chọn không hợp lệ, thử lại.")
