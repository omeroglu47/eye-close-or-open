import cv2
import dlib

detector=dlib.get_frontal_face_detector()
model=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

def mid(p1,p2):
    return (int((p1[0]+p2[0])/2),   int((p1[1]+p2[1])/2))    # değerler float çizimde innt lazım

f=open("dataset.csv","a")

while True:
    _,frame=cap.read()
    frame=cv2.flip(frame,1) # 180 derece aynaladık
    gri=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces=detector(frame)

    for face in faces:
        points=model(gri,face)

        points_list=[(p.x,p.y) for p in points.parts()]
        #rint(points_list)

        p1,p2=points_list[37],points_list[38] # üst göz noktaları sol
        p3, p4 = points_list[40], points_list[41] # alt göz noktaları sol

        p5,p6=points_list[43],points_list[44] # üst göz noktaları sag
        p7, p8 = points_list[46], points_list[47] # alt göz noktaları sag


        #cv2.circle(frame,(p1[0],p1[1]),3,(0,0,255),-1)
        #cv2.circle(frame,(p2[0], p2[1]), 3, (0, 0, 255), -1)
        #cv2.circle(frame,(p3[0],p3[1]),3,(0,0,255),-1)
        #cv2.circle(frame,(p4[0], p4[1]), 3, (0, 0, 255), -1)

        #sol göz
        po_ust_sol=mid(p1,p2)    # değer gelicek x,y
        po_alt_sol=mid(p3,p4)

        sol_mesafe = po_alt_sol[1] - po_ust_sol[1]

        cv2.circle(frame, (po_ust_sol[0], po_ust_sol[1]), 3, (0, 255, 0), -1)
        cv2.circle(frame, (po_alt_sol[0], po_alt_sol[1]), 3, (0, 255, 0), -1)

        #sag göz
        po_ust_sag=mid(p5,p6)    # değer gelicek x,y
        po_alt_sag=mid(p7,p8)
        sag_mesafe = po_alt_sag[1] - po_ust_sag[1] # orta noktalar ar. mesafe





        cv2.circle(frame, (po_ust_sag[0], po_ust_sag[1]), 3, (0, 255, 0), -1)
        cv2.circle(frame, (po_alt_sag[0], po_alt_sag[1]), 3, (0, 255, 0), -1)




        #burun
        mburun=points_list[30][1]-points_list[27][1]     #27,28,29,30 burun noktaları
        print("burun :::",mburun)

        print("sol_göz:",po_alt_sol[1]-po_ust_sol[1])
        print("sag_göz:",po_alt_sag[1]-po_ust_sag   [1])
        #bu verileri dosyaya yazdırmalıyım


        # üst göz işaretlemeleri tamam şimdi alt göze iniyruz
        # noktalı modelin jpg i ni açtım ve ilgi noktalarım 41 ,40






    cv2.imshow("sd",frame)

    if cv2.waitKey(1) & 0xFF==ord("q"):
        r=input("işlem girin :")

        if r=="q":
            break
        elif r=="v":

            # sol ve sağ değerlerini yazma sebebimiz veri dateset oluşturmak için
            # gözün kapalı ve aık olduğu durumları ve gerekli konumları 20-30 arası
            # veri kayıt edilerek veri setini eğiteceğiz
            sol=input("sol göz için veri girin:")
            sag = input("sağ göz için veri girin:")

            #print(f"sol göz: {sol_mesafe} sağ göz: {sag_mesafe} burun : {mburun}")

            f.write(f"{sol_mesafe},{sag_mesafe},{mburun},{sol},{sag}\n")





cap.release()
cv2.destroyAllWindows()