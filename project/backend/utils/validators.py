"""
utils/validators.py
New datasets:
  Diabetes: gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level
  Heart/Cardio: age_years,gender,height,weight,bmi,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active
"""
DIABETES_FIELDS = ['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']
DIABETES_BOUNDS = {'gender':(0,2),'age':(0,90),'hypertension':(0,1),'heart_disease':(0,1),'smoking_history':(0,5),'bmi':(10,100),'HbA1c_level':(3,15),'blood_glucose_level':(50,500)}

HEART_FIELDS = ['age_years','gender','height','weight','bmi','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']
HEART_BOUNDS = {'age_years':(1,100),'gender':(1,2),'height':(100,220),'weight':(30,200),'bmi':(10,60),'ap_hi':(60,250),'ap_lo':(40,200),'cholesterol':(1,3),'gluc':(1,3),'smoke':(0,1),'alco':(0,1),'active':(0,1)}

def validate_fields(data,fields,bounds):
    missing=[f for f in fields if f not in data]
    if missing: return None,f"Missing: {', '.join(missing)}"
    values=[]
    for f in fields:
        try: values.append(float(data[f]))
        except: return None,f"Field '{f}' must be numeric"
    for f,v in zip(fields,values):
        lo,hi=bounds[f]
        if not(lo<=v<=hi): return None,f"Field '{f}'={v} outside [{lo},{hi}]"
    return values,None
