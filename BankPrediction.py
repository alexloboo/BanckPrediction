import tkinter as tk
from tkinter import Entry, LabelFrame
from tkinter import messagebox
from tkinter import ttk
from tkinter.constants import COMMAND, END, VERTICAL
#--------------------------generación modelo, predicción
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pickle
import time

class MyApp(tk.Tk):
    def __init__(self, *args, **kwargs, ):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        self.frames = {}
        # --------------------------- menu
        menu = tk.Menu(container)
        betting = tk.Menu(menu, tearoff=0)
        menu.add_cascade(menu=betting, label="Opciones")
        betting.add_command(label="Modelo Test",command=lambda: self.show_frame(Startpage))
        betting.add_command(label="Predicción",command=lambda: self.show_frame(PagePrediction))
        
        tk.Tk.config(self, menu=menu)

        for F in (Startpage, PagePrediction):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            frame.config(bg="#3CB371")

        self.show_frame(Startpage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

class Startpage(tk.Frame):  
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        
        tt = tk.Label(self, text="BIENVENIDO A BANKPREDICTOR",bg="#3CB371",fg='white',font=('Helvetica', 18, 'bold')).place(x=170,y=35)
        div = tk.Label(self, text="_______________________________________________________",bg="#3CB371",fg='black',font=('Helvetica', 15, 'bold')).place(x=50,y=110)
        l1 = tk.Label(self, text="BANKPREDICTOR busca predecir un el cliente se suscribirá \n\na un depósito a plazo (term deposit).",bg="#3CB371",fg='black')
        l1.place(x=190,y=110)
        div2 = tk.Label(self, text="_______________________________________________________",bg="#3CB371",fg='black',font=('Helvetica', 15, 'bold')).place(x=50,y=400)
        
        t1 = tk.Label(self, text="OBJETIVOS",bg="#3CB371",fg='black',font=('Helvetica', 12, 'bold')).place(x=312,y=190)
        t2 = tk.Label(self, text=" Neuronas (x)                ",bg="#3CB371",fg='black',font=('Helvetica', 8, 'italic')).place(x=320,y=280)
        t3 = tk.Label(self, text="Se va generar un modelo con el menor error posible del rango 5 a 'x'. \nInsertar en el Entry la cantidad maxima de neuroas en el hidden layer a analizar",bg="#3CB371",fg='black').place(x=170,y=240)
        #--------------------------- Entry
        self.e_nhl = tk.Entry(self,bg="#d9dcd6",fg="#16425B", width=12)
        self.e_nhl.place(x=320,y=300)

        # -------------------------- generar modelo
        act = tk.Button(self, text="Generar Modelo",height = 1, width = 14,bg="#fedc56",command= self.gModelo)
        act.place(x=250,y=330)
        # -------------------------- test
        act = tk.Button(self, text="Crear Test",height = 1, width = 14,bg="#fedc56",command= self.gTest)
        act.place(x=370,y=330)

        #------------------------------------aviso legal
        w = tk.Canvas(self, width=610, bg="#3CB371",height=100).place(x=50,y=460)
        e = tk.Label(self, text="AVISO LEGAL",bg="#3CB371",fg='black',font=('Helvetica', 12, 'bold')).place(x=300,y=465)
        sub1 = tk.Label(self, text=" Bankpredictor excluye cualquier responsabilidad por los daños y perjuicios de toda naturaleza que pudieran deberse\n a la mala utilización del Servicio. Por motivos de seguridad, Bankprediction no recopila\n ni almacena información confidencial de su empresa",bg="#3CB371",fg='black',font=('Helvetica', 8, 'italic')).place(x=60,y=500)

    def gModelo(self):
        ehl = self.e_nhl.get()
        if (ehl == ""):
            messagebox.showinfo("Estatus", "Se requiere insertar la cantidad máxima de neuronas (x), porque se va buscar el menor error de 5 a x neuronas en hidden layer")
        else:
            messagebox.showinfo("Estatus", "Generando el modelo con la menor catidad de error del rango de neuronas de hl.")
            # leyendo el csv sacando los datos de X y Y
            x = np.loadtxt('train.csv', delimiter=',', usecols=range(16))
            y = np.loadtxt('train.csv', delimiter=',', usecols=(16, ))
                
            # se construyen conjuntos de entrenamiento y prueba al azar
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=3)
            eTupple =[]
            # hlmax = int(input('Se buscará el menor error desde el hidden layer 5 hasta: '))
            for hl in range(5,int(ehl)):
                # se usa el conjunto de prueba para entrenar el clasificador
                clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(hl,), random_state=1)
                clf.fit(x_train, y_train)

                # se usa el modelo entrenado para predecir las salidas sobre el conjunto de prueba
                predicted = clf.predict(x_test)

                # se calcula el error sobre el conjunto de prueba
                error = 1 - accuracy_score(y_test, predicted)
                eTupple.append((error,hl))
                
            hl = min(eTupple)
            # print(eTupple)
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(hl[1],), random_state=1)
            clf.fit(x_train, y_train)

            #Resultados
            predicted = clf.predict(x_test)
            print('\n--------BANK-------\nErrores de diferentes cantidades de neuronas en hl (error,#hl): {}\n'.format((eTupple)))
            messagebox.showinfo("Estatus", 'Error menor sobre el conjunto de prueba: {}, HL: {}'.format(hl[0],hl[1]))
            messagebox.showinfo("Estatus", 'En la consola se mostrará el reporte de clasificación y la matrix de confusión')
            print('Reporte de clasificación: \n',metrics.classification_report(y_test, predicted))
            print('Matrix de confusión: \n',metrics.confusion_matrix(y_test, predicted))

            # se guarda el modelo entrenado para uso posterior
            filename = 'trained_modelBankMLP.sav'
            pickle.dump(clf, open(filename, 'wb'))
            self.e_nhl.delete(0,'end')    
        pass
    
    def gTest(self):    
        messagebox.showinfo("Estatus", "Sacando el error del archivo Test con 4521 ejemplos de datos con el modelo")
        x = np.loadtxt('test.csv', delimiter=',', usecols=range(16))
        y = np.loadtxt('test.csv', delimiter=',', usecols=(16, ))

        # se carga la red neuronal entrenada
        clf = pickle.load(open('trained_modelBankMLP.sav', 'rb'))

        # se usa el modelo entrenado para predecir las salidas 
        predicted = clf.predict(x)

        # se calcula el error de prediccion
        error = 1 - accuracy_score(y, predicted)
        messagebox.showinfo("Estatus", 'Error de prediccion del test: {}'.format(error))
        pass

class PagePrediction(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        ll = tk.Label(self, text="PREDICCIÓN",bg="#3CB371",fg='white',font=('Helvetica', 25, 'bold')).place(x=250,y=35)
        
        div = tk.Label(self, text="_______________________________________________________",bg="#3CB371",fg='black',font=('Helvetica', 15, 'bold')).place(x=50,y=110)
        div2 = tk.Label(self, text="_______________________________________________________",bg="#3CB371",fg='black',font=('Helvetica', 15, 'bold')).place(x=50,y=400)
        
        
        sp_op = tk.Label(self, text="Edad",bg="#3CB371",fg='black').place(x=70,y=170)
        sp_op = tk.Label(self, text="Trabajo",bg="#3CB371",fg='black').place(x=70,y=200)
        sp_op = tk.Label(self, text="Estado Civil",bg="#3CB371",fg='black').place(x=70,y=230)
        sp_op = tk.Label(self, text="Educación",bg="#3CB371",fg='black').place(x=70,y=260)

        sp_op = tk.Label(self, text="¿Cuenta con tarjeta de credito?",bg="#3CB371",fg='black').place(x=70,y=290)
        sp_op = tk.Label(self, text="Salario promedio anual",bg="#3CB371",fg='black').place(x=70,y=320)
        sp_op = tk.Label(self, text="¿Tiene préstamo para vivienda?",bg="#3CB371",fg='black').place(x=70,y=350)
        sp_op = tk.Label(self, text="¿Tiene préstamo personal?",bg="#3CB371",fg='black').place(x=70,y=380)

        self.p1 = tk.Entry(self,bg="#d9dcd6",fg="#16425B", width=12)
        self.p1.place(x=240,y=170)
        self.p2 = tk.Entry(self,bg="#d9dcd6",fg="#16425B", width=12)
        self.p2.place(x=240,y=200)
        self.p3 = tk.Entry(self,bg="#d9dcd6",fg="#16425B", width=12)
        self.p3.place(x=240,y=230)
        self.p4 = tk.Entry(self,bg="#d9dcd6",fg="#16425B", width=12)
        self.p4.place(x=240,y=260)

        self.p5 = tk.Entry(self,bg="#d9dcd6",fg="#16425B", width=12)
        self.p5.place(x=240,y=290)
        self.p6 = tk.Entry(self,bg="#d9dcd6",fg="#16425B", width=12)
        self.p6.place(x=240,y=320)
        self.p7 = tk.Entry(self,bg="#d9dcd6",fg="#16425B", width=12)
        self.p7.place(x=240,y=350)
        self.p8 = tk.Entry(self,bg="#d9dcd6",fg="#16425B", width=12)
        self.p8.place(x=240,y=380)

        sp_op = tk.Label(self, text="Tipo de comunicación",bg="#3CB371",fg='black').place(x=380,y=170)
        sp_op = tk.Label(self, text="Último dia de comunicación",bg="#3CB371",fg='black').place(x=380,y=200)
        sp_op = tk.Label(self, text="Último mes de comunicación",bg="#3CB371",fg='black').place(x=380,y=230)
        sp_op = tk.Label(self, text="Duración",bg="#3CB371",fg='black').place(x=380,y=260)

        sp_op = tk.Label(self, text="Cantidad de campaña",bg="#3CB371",fg='black').place(x=380,y=290)
        sp_op = tk.Label(self, text="Días desde la última com.",bg="#3CB371",fg='black').place(x=380,y=320)
        sp_op = tk.Label(self, text="Cantidad de comunicación\n antes de la campaña",bg="#3CB371",fg='black').place(x=380,y=350)
        sp_op = tk.Label(self, text="Resultado Marketing anterior",bg="#3CB371",fg='black').place(x=380,y=380)

        self.p9 = tk.Entry(self,bg="#d9dcd6",fg="#16425B", width=12)
        self.p9.place(x=555,y=170)
        self.p10 = tk.Entry(self,bg="#d9dcd6",fg="#16425B", width=12)
        self.p10.place(x=555,y=200)
        self.p11 = tk.Entry(self,bg="#d9dcd6",fg="#16425B", width=12)
        self.p11.place(x=555,y=230)
        self.p12 = tk.Entry(self,bg="#d9dcd6",fg="#16425B", width=12)
        self.p12.place(x=555,y=260)

        self.p13 = tk.Entry(self,bg="#d9dcd6",fg="#16425B", width=12)
        self.p13.place(x=555,y=290)
        self.p14 = tk.Entry(self,bg="#d9dcd6",fg="#16425B", width=12)
        self.p14.place(x=555,y=320)
        self.p15 = tk.Entry(self,bg="#d9dcd6",fg="#16425B", width=12)
        self.p15.place(x=555,y=350)
        self.p16 = tk.Entry(self,bg="#d9dcd6",fg="#16425B", width=12)
        self.p16.place(x=555,y=380)

        e_pp = tk.Label(self, text="Predicción: ",bg="#3CB371",fg='black').place(x=70,y=440)

        self.pp = tk.Entry(self,bg="#d9dcd6",fg="#16425B", width=12)
        self.pp.place(x=240,y=440)

        self.btn_p = tk.Button(self, text="Crear predicción",bg="#fedc56",
                        command=lambda: [self.pprediction()])
        self.btn_p.place(x=534,y=440)

        self.btn_p = tk.Button(self, text="Vaciar campos",bg="#CD5C5C",
                        command=lambda: [self.vc()])
        self.btn_p.place(x=430,y=440)
        pass
    
    def pprediction(self):
        p1 = self.p1.get()
        p2 = self.p2.get()
        p3 = self.p3.get()
        p4 = self.p4.get()
        p5 = self.p5.get()
        p6 = self.p6.get()
        p7 = self.p7.get()
        p8 = self.p8.get()

        p9 = self.p9.get()
        p10 = self.p10.get()
        p11 = self.p11.get()
        p12 = self.p12.get()
        p13 = self.p13.get()
        p14 = self.p14.get()
        p15 = self.p15.get()
        p16 = self.p16.get()
        if (p1 == "" or p2 == "" or p3 == "" or p4 == "" or p5 == "" or p6 == "" or p7 == "" or p8 == "" or p9 == "" or p10 == "" or p11 == "" or p12 == "" or p13 == "" or p14 == "" or p15 == "" or p16 == ""):
            messagebox.showinfo("Estatus", "Se requiere llenar todos los campos para hacer una predicción")
        else:
            pred = []
            pred.append(int(p1))
            pred.append(int(p2))
            pred.append(int(p3))
            pred.append(int(p4))
            pred.append(int(p5))
            pred.append(int(p6))
            pred.append(int(p7))
            pred.append(int(p8))
            pred.append(int(p9))
            pred.append(int(p10))
            pred.append(int(p11))
            pred.append(int(p12))
            pred.append(int(p13))
            pred.append(int(p14))
            pred.append(int(p15))
            pred.append(int(p16))

            x = np.loadtxt('test.csv', delimiter=',', usecols=range(16))
            y = np.loadtxt('test.csv', delimiter=',', usecols=(16, ))

            # se carga la red neuronal entrenada
            clf = pickle.load(open('trained_modelBankMLP.sav', 'rb'))

            # se usa el modelo entrenado para predecir las salidas 
            predicted = clf.predict(x)

            #------------------predicción
            Xnew = [pred]
            ynew = clf.predict(Xnew)
            ynew=ynew[0]
            print("predicción: ",(ynew))
            if ynew == 2:
                self.pp.config(state=tk.NORMAL)
                self.pp.insert(0, "NO")
            else:
                self.pp.config(state=tk.NORMAL)
                self.pp.insert(0, "YES")
        pass

    def vc(self):
        self.p1.delete(0,'end')
        self.p2.delete(0,'end')
        self.p3.delete(0,'end')
        self.p4.delete(0,'end')
        self.p5.delete(0,'end')
        self.p6.delete(0,'end')
        self.p7.delete(0,'end')
        self.p8.delete(0,'end')
        self.p9.delete(0,'end')
        self.p10.delete(0,'end')
        self.p11.delete(0,'end')
        self.p12.delete(0,'end')
        self.p13.delete(0,'end')
        self.p14.delete(0,'end')
        self.p15.delete(0,'end')
        self.p16.delete(0,'end')  
        self.p16.delete(0,'end')  
        self.pp.delete(0,'end')
        pass

#-------------configuración basica de la ventanda
app = MyApp()
app.geometry("720x600")
app.title("   BANKPREDICTOR")
app.resizable(False,False)
app.mainloop()
