from rouge import Rouge
import rouge


tp_data= open("./Summaries/system1_Sample doc.txt","r")
e_sum_data=open("./Summaries/system3_Sample doc.txt","r")
tr_data=open("./Summaries/system2_Sample doc.txt","r")
main_data_1=tp_data.read()
main_data_2=e_sum_data.read()
main_data_3=tr_data.read()
rouge=Rouge()
result1=rouge.get_scores(main_data_1,main_data_2)
result2=rouge.get_scores(main_data_2,main_data_3)
print("\n The Result Of TP And E-Sum Is :")
for val in result1:
    for data in val.values():
        for digit in data.values():
            print(round(digit*100,2),"%")
print(result1)
print("\n The Result Of TR And E-Sum Is :")
for val in result2:
    for data in val.values():
        for digit in data.values():
            print(round(digit*100,2),"%")

print(result2)