{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "aa70d504",
   "metadata": {
    "papermill": {
     "duration": 0.035072,
     "end_time": "2024-03-10T10:04:38.582831",
     "exception": false,
     "start_time": "2024-03-10T10:04:38.547759",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<img src = \"https://seeklogo.com/images/S/streamlit-logo-1A3B208AE4-seeklogo.com.png\"\n",
    "     style=\"  display: block;\n",
    "  margin-left: auto;\n",
    "  margin-right: auto;\n",
    "  width: 18%;\">\n",
    "<h2 style= \"background-color: #000; \n",
    "            padding: 15px; \n",
    "            font: bold 26px tahoma;\n",
    "            text-align: center; \n",
    "            color:#FF4848;\n",
    "            border: 6px solid red;\n",
    "            border-radius: 7px;\">   \n",
    "    🚀📊 The Link To My <u>Streamlit Web App</u> For This Project In The Last Section Below 🚀🤩\n",
    "</h2>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "abefecd1",
   "metadata": {
    "papermill": {
     "duration": 0.033473,
     "end_time": "2024-03-10T10:04:38.651711",
     "exception": false,
     "start_time": "2024-03-10T10:04:38.618238",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Importing Toolkits💼🔨"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "31dfcd73",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:38.723901Z",
     "iopub.status.busy": "2024-03-10T10:04:38.723084Z",
     "iopub.status.idle": "2024-03-10T10:04:42.340864Z",
     "shell.execute_reply": "2024-03-10T10:04:42.339516Z"
    },
    "papermill": {
     "duration": 3.657782,
     "end_time": "2024-03-10T10:04:42.344393",
     "exception": false,
     "start_time": "2024-03-10T10:04:38.686611",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import plotly.express as px\n",
    "from plotly.offline import iplot\n",
    "\n",
    "from xgboost import XGBClassifier\n",
    "\n",
    "from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold, cross_val_score\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "# pd.set_option('future.no_silent_downcasting', True)\n",
    "pd.options.mode.copy_on_write = \"warn\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "989eb53a",
   "metadata": {
    "papermill": {
     "duration": 0.036225,
     "end_time": "2024-03-10T10:04:42.416904",
     "exception": false,
     "start_time": "2024-03-10T10:04:42.380679",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### Customer Vizualization Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "21b7da91",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:42.492842Z",
     "iopub.status.busy": "2024-03-10T10:04:42.491662Z",
     "iopub.status.idle": "2024-03-10T10:04:42.502480Z",
     "shell.execute_reply": "2024-03-10T10:04:42.500786Z"
    },
    "papermill": {
     "duration": 0.052523,
     "end_time": "2024-03-10T10:04:42.505372",
     "exception": false,
     "start_time": "2024-03-10T10:04:42.452849",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Adding Line To Plotly Figure\n",
    "def add_line(x0 = 0, y0 = 0, x1 = 0, y1 = 0, \n",
    "             line_color = \"#00DFA2\", font_color = \"#3C486B\", \n",
    "             xposition = \"right\", text = \"Text\"):\n",
    "    fig.add_shape(type='line',\n",
    "                  x0 = x0,\n",
    "                  y0 = y0,\n",
    "                  x1 = x1,\n",
    "                  y1 = y1 + 2,\n",
    "                  line = {\n",
    "                      \"color\" : line_color,\n",
    "                      \"width\" : 3,\n",
    "                      \"dash\" : \"dashdot\"\n",
    "                  },\n",
    "                  label={\n",
    "                      \"text\" : f\"\\t{text}: {x1: 0.1f}\\t\".expandtabs(5),\n",
    "                      \"textposition\": \"end\",\n",
    "                      \"yanchor\" :\"top\",\n",
    "                      \"xanchor\" :xposition,\n",
    "                      \"textangle\" :0,\n",
    "                      \"font\": {\n",
    "                          \"size\": 14,\n",
    "                          \"color\" :font_color,\n",
    "                          \"family\" : \"arial\"\n",
    "\n",
    "                      },\n",
    "                  }\n",
    "                 )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "3d9e2623",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:42.578297Z",
     "iopub.status.busy": "2024-03-10T10:04:42.577769Z",
     "iopub.status.idle": "2024-03-10T10:04:42.585303Z",
     "shell.execute_reply": "2024-03-10T10:04:42.583817Z"
    },
    "papermill": {
     "duration": 0.047429,
     "end_time": "2024-03-10T10:04:42.588130",
     "exception": false,
     "start_time": "2024-03-10T10:04:42.540701",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def custome_layout(title_size = 28, hover_font_size = 16, showlegend = False):\n",
    "    fig.update_layout(\n",
    "    showlegend = showlegend,\n",
    "    title = {\n",
    "        \"font\" :{\n",
    "            \"size\" :title_size,\n",
    "            \"family\" : \"tahoma\"\n",
    "        }\n",
    "    },\n",
    "    hoverlabel = {\n",
    "        \"bgcolor\" :\"#111\",\n",
    "        \"font_size\" : hover_font_size,\n",
    "        \"font_family\" :\"arial\"\n",
    "    }\n",
    "\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "08b2bb52",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:42.660755Z",
     "iopub.status.busy": "2024-03-10T10:04:42.660291Z",
     "iopub.status.idle": "2024-03-10T10:04:42.671271Z",
     "shell.execute_reply": "2024-03-10T10:04:42.670232Z"
    },
    "papermill": {
     "duration": 0.050907,
     "end_time": "2024-03-10T10:04:42.673662",
     "exception": false,
     "start_time": "2024-03-10T10:04:42.622755",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def count_bar_plot(data_frame, column_name, x_title= None, y_title= None, title=\"None\", title_size = 28, \n",
    "                   showlegend = False, hover_template=\"None\", bars_names = None):\n",
    "    \n",
    "    counts = data_frame[column_name].value_counts(normalize = 1) * 100\n",
    "    \n",
    "    fig = px.bar(\n",
    "            data_frame=counts,\n",
    "            x = counts.index if bars_names is None else bars_names,\n",
    "            y = counts / sum(counts) * 100,\n",
    "            template=\"plotly_white\",\n",
    "            labels={\"x\": x_title if x_title is not None else column_name, \"y\" :\"Frequency in PCT(%)\"},\n",
    "            text= counts.apply(lambda x: f\"{x:0.0f}%\"),\n",
    "            title=title,\n",
    "            color=counts.index.astype(str)\n",
    "\n",
    "        )       \n",
    "\n",
    "    fig.update_layout(\n",
    "        showlegend = showlegend,\n",
    "        title = {\n",
    "            \"font\" :{\n",
    "                \"size\" :title_size,\n",
    "                \"family\" : \"tahoma\"\n",
    "            }\n",
    "        },\n",
    "        hoverlabel = {\n",
    "            \"bgcolor\" :\"#111\",\n",
    "            \"font_size\" : 16,\n",
    "            \"font_family\" :\"arial\"\n",
    "        }\n",
    "    )\n",
    "\n",
    "    fig.update_traces(\n",
    "        textfont = {\n",
    "            \"size\" : 18,\n",
    "            \"family\" :\"consolas\",\n",
    "            \"color\": \"#fff\"\n",
    "        },\n",
    "        hovertemplate = hover_template,\n",
    "    )\n",
    "\n",
    "    return fig"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e7efc53d",
   "metadata": {
    "papermill": {
     "duration": 0.035716,
     "end_time": "2024-03-10T10:04:42.746059",
     "exception": false,
     "start_time": "2024-03-10T10:04:42.710343",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Loading Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "12868e73",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:42.819363Z",
     "iopub.status.busy": "2024-03-10T10:04:42.818295Z",
     "iopub.status.idle": "2024-03-10T10:04:42.851786Z",
     "shell.execute_reply": "2024-03-10T10:04:42.850268Z"
    },
    "papermill": {
     "duration": 0.073876,
     "end_time": "2024-03-10T10:04:42.854993",
     "exception": false,
     "start_time": "2024-03-10T10:04:42.781117",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"/kaggle/input/heart-failure-prediction/heart.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b45fd743",
   "metadata": {
    "_kg_hide-input": false,
    "_kg_hide-output": true,
    "papermill": {
     "duration": 0.034881,
     "end_time": "2024-03-10T10:04:42.925085",
     "exception": false,
     "start_time": "2024-03-10T10:04:42.890204",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Columns Details 💊\n",
    "<ol style = \"font: bold 15px arial; color: #222; background-color:#fff; margin:30px; padding: 30px\">\t\n",
    "    <li style = \"margin-bottom: 10px\">Age → The age of the patient in years.</li>\n",
    "    <hr>\n",
    "    <li style = \"margin-bottom: 10px\">Sex → The Gender of the patient.\n",
    "         <ul>\n",
    "            <li>F: Female</li>\n",
    "            <li>M: Male</li>\n",
    "         </ul>\n",
    "    </li>\n",
    "    <hr>\n",
    "    <li style = \"margin-bottom: 10px\">ChestPainType → the type of chest pain experienced by the patient:\n",
    "         <ul>\n",
    "            <li>TA: Typical angina (chest pain related to decreased blood supply to the heart).</li>\n",
    "            <li>ATA: Atypical angina (chest pain <u>not</u> directly related to the heart).</li>\n",
    "            <li>NAP: Non-anginal pain (usually esophageal spasms).</li>\n",
    "            <li>ASY: Asymptomatic (no symptoms).</li>\n",
    "         </ul>\n",
    "    </li>\n",
    "    <hr>\n",
    "    <li style = \"margin-bottom: 10px\">\n",
    "        RestingBP (Resting Blood Pressure) → The patient's resting blood pressure (in mmHg on admission to the hospital).\n",
    "    </li>\n",
    "    <hr>\n",
    "    <li style = \"margin-bottom: 10px\">Cholesterol → The patient's cholesterol measurement in mg/dl.</li>\n",
    "    <hr>\n",
    "    <li style = \"margin-bottom: 10px\">\n",
    "        FastingBS (Fasting Blood Sugar) → This column indicates whether the patient's fasting blood sugar is greater than 120 mg/dl\n",
    "         <ul>\n",
    "            <li>1 (true): if the fasting blood sugar is greater than 120 mg/dl</li>\n",
    "            <li>0 (false) otherwise</li>\n",
    "         </ul>\n",
    "    </li>\n",
    "    <hr>\n",
    "    <li style = \"margin-bottom: 10px\">\n",
    "        RestingECG (Resting ElectroCardioGraphic) → shows the results of an ECG (electrocardiogram) test that measures the heart's electrical activity\n",
    "         <ul>\n",
    "            <li>Normal: Normal</li>\n",
    "            <li>ST: Having ST-T wave abnormalities (indications of minor irregularities in the heart's electrical activity)</li>\n",
    "            <li>LVH: Showing probable or definite \"Left Ventricular Hypertrophy\" by Estes' criteria</li>\n",
    "         </ul>\n",
    "    </li>\n",
    "    <hr>\n",
    "    <li style = \"margin-bottom: 10px\">MaxHR (Maximum Heart Rate) → The highest heart rate achieved by the patient during a stress test.</li>\n",
    "    <hr>\n",
    "    <li style = \"margin-bottom: 10px\">ExerciseAngina (Exercise Angina) → This column indicates whether the patient experienced angina (chest pain) during exercise. \n",
    "         <ul>\n",
    "            <li>Y (Yes): angina was induced by exercise</li>\n",
    "            <li>N (No): otherwise</li>\n",
    "         </ul>\n",
    "    </li>\n",
    "    <hr>\n",
    "    <li style = \"margin-bottom: 10px\">Oldpeak (ST depression induced by exercise relative to rest) → Measures the amount of ST depression (finding on an ECG) during peak exercise compared to rest\n",
    "    </li>\n",
    "    <hr>\n",
    "    <li style = \"margin-bottom: 10px\">ST_Slope (the Slope of the peak exercise ST segment) → Indicates the slope of the peak exercise ST segment\n",
    "         <ul>\n",
    "            <li>Up: Upsloping</li>\n",
    "            <li>Flat: Flat</li>\n",
    "            <li>Down: Downsloping</li>\n",
    "         </ul>\n",
    "    </li>\n",
    "    <hr>\n",
    "    <li style = \"margin-bottom: 10px\">HeartDisease (Our Target) \n",
    "         <ul>\n",
    "            <li>0</li>\n",
    "            <li>1</li>\n",
    "         </ul>\n",
    "    </li>            \n",
    "</ol>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fc1843a8",
   "metadata": {
    "papermill": {
     "duration": 0.128198,
     "end_time": "2024-03-10T10:04:43.088904",
     "exception": false,
     "start_time": "2024-03-10T10:04:42.960706",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Let's Get Quick Overview 🧐😉"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "88cdc8e0",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:43.161279Z",
     "iopub.status.busy": "2024-03-10T10:04:43.160476Z",
     "iopub.status.idle": "2024-03-10T10:04:43.165889Z",
     "shell.execute_reply": "2024-03-10T10:04:43.165036Z"
    },
    "papermill": {
     "duration": 0.044613,
     "end_time": "2024-03-10T10:04:43.168433",
     "exception": false,
     "start_time": "2024-03-10T10:04:43.123820",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "• Number of Records: 918\n",
      "• Number of Features: 12\n"
     ]
    }
   ],
   "source": [
    "print(f\"• Number of Records: {df.shape[0]:,.0f}\")\n",
    "print(f\"• Number of Features: {df.shape[1]}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "38c4337e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:43.241822Z",
     "iopub.status.busy": "2024-03-10T10:04:43.241056Z",
     "iopub.status.idle": "2024-03-10T10:04:43.273640Z",
     "shell.execute_reply": "2024-03-10T10:04:43.272789Z"
    },
    "papermill": {
     "duration": 0.072791,
     "end_time": "2024-03-10T10:04:43.276333",
     "exception": false,
     "start_time": "2024-03-10T10:04:43.203542",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 918 entries, 0 to 917\n",
      "Data columns (total 12 columns):\n",
      " #   Column          Non-Null Count  Dtype  \n",
      "---  ------          --------------  -----  \n",
      " 0   Age             918 non-null    int64  \n",
      " 1   Sex             918 non-null    object \n",
      " 2   ChestPainType   918 non-null    object \n",
      " 3   RestingBP       918 non-null    int64  \n",
      " 4   Cholesterol     918 non-null    int64  \n",
      " 5   FastingBS       918 non-null    int64  \n",
      " 6   RestingECG      918 non-null    object \n",
      " 7   MaxHR           918 non-null    int64  \n",
      " 8   ExerciseAngina  918 non-null    object \n",
      " 9   Oldpeak         918 non-null    float64\n",
      " 10  ST_Slope        918 non-null    object \n",
      " 11  HeartDisease    918 non-null    int64  \n",
      "dtypes: float64(1), int64(6), object(5)\n",
      "memory usage: 86.2+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "0a16ad49",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:43.351225Z",
     "iopub.status.busy": "2024-03-10T10:04:43.350476Z",
     "iopub.status.idle": "2024-03-10T10:04:43.385087Z",
     "shell.execute_reply": "2024-03-10T10:04:43.383789Z"
    },
    "papermill": {
     "duration": 0.075866,
     "end_time": "2024-03-10T10:04:43.388257",
     "exception": false,
     "start_time": "2024-03-10T10:04:43.312391",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Sex</th>\n",
       "      <th>ChestPainType</th>\n",
       "      <th>RestingBP</th>\n",
       "      <th>Cholesterol</th>\n",
       "      <th>FastingBS</th>\n",
       "      <th>RestingECG</th>\n",
       "      <th>MaxHR</th>\n",
       "      <th>ExerciseAngina</th>\n",
       "      <th>Oldpeak</th>\n",
       "      <th>ST_Slope</th>\n",
       "      <th>HeartDisease</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>840</th>\n",
       "      <td>41</td>\n",
       "      <td>M</td>\n",
       "      <td>ATA</td>\n",
       "      <td>135</td>\n",
       "      <td>203</td>\n",
       "      <td>0</td>\n",
       "      <td>Normal</td>\n",
       "      <td>132</td>\n",
       "      <td>N</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Flat</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>681</th>\n",
       "      <td>51</td>\n",
       "      <td>M</td>\n",
       "      <td>ASY</td>\n",
       "      <td>140</td>\n",
       "      <td>261</td>\n",
       "      <td>0</td>\n",
       "      <td>LVH</td>\n",
       "      <td>186</td>\n",
       "      <td>Y</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Up</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>395</th>\n",
       "      <td>38</td>\n",
       "      <td>M</td>\n",
       "      <td>ASY</td>\n",
       "      <td>135</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>Normal</td>\n",
       "      <td>150</td>\n",
       "      <td>N</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Flat</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>129</th>\n",
       "      <td>42</td>\n",
       "      <td>M</td>\n",
       "      <td>NAP</td>\n",
       "      <td>120</td>\n",
       "      <td>228</td>\n",
       "      <td>0</td>\n",
       "      <td>Normal</td>\n",
       "      <td>152</td>\n",
       "      <td>Y</td>\n",
       "      <td>1.5</td>\n",
       "      <td>Flat</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>793</th>\n",
       "      <td>67</td>\n",
       "      <td>M</td>\n",
       "      <td>ASY</td>\n",
       "      <td>125</td>\n",
       "      <td>254</td>\n",
       "      <td>1</td>\n",
       "      <td>Normal</td>\n",
       "      <td>163</td>\n",
       "      <td>N</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Flat</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>569</th>\n",
       "      <td>55</td>\n",
       "      <td>M</td>\n",
       "      <td>ASY</td>\n",
       "      <td>158</td>\n",
       "      <td>217</td>\n",
       "      <td>0</td>\n",
       "      <td>Normal</td>\n",
       "      <td>110</td>\n",
       "      <td>Y</td>\n",
       "      <td>2.5</td>\n",
       "      <td>Flat</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>831</th>\n",
       "      <td>63</td>\n",
       "      <td>F</td>\n",
       "      <td>NAP</td>\n",
       "      <td>135</td>\n",
       "      <td>252</td>\n",
       "      <td>0</td>\n",
       "      <td>LVH</td>\n",
       "      <td>172</td>\n",
       "      <td>N</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Up</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>650</th>\n",
       "      <td>48</td>\n",
       "      <td>M</td>\n",
       "      <td>ASY</td>\n",
       "      <td>130</td>\n",
       "      <td>256</td>\n",
       "      <td>1</td>\n",
       "      <td>LVH</td>\n",
       "      <td>150</td>\n",
       "      <td>Y</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Up</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>375</th>\n",
       "      <td>73</td>\n",
       "      <td>F</td>\n",
       "      <td>NAP</td>\n",
       "      <td>160</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>ST</td>\n",
       "      <td>121</td>\n",
       "      <td>N</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Up</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>331</th>\n",
       "      <td>56</td>\n",
       "      <td>M</td>\n",
       "      <td>ASY</td>\n",
       "      <td>115</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>ST</td>\n",
       "      <td>82</td>\n",
       "      <td>N</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>Up</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Age Sex ChestPainType  RestingBP  Cholesterol  FastingBS RestingECG  \\\n",
       "840   41   M           ATA        135          203          0     Normal   \n",
       "681   51   M           ASY        140          261          0        LVH   \n",
       "395   38   M           ASY        135            0          1     Normal   \n",
       "129   42   M           NAP        120          228          0     Normal   \n",
       "793   67   M           ASY        125          254          1     Normal   \n",
       "569   55   M           ASY        158          217          0     Normal   \n",
       "831   63   F           NAP        135          252          0        LVH   \n",
       "650   48   M           ASY        130          256          1        LVH   \n",
       "375   73   F           NAP        160            0          0         ST   \n",
       "331   56   M           ASY        115            0          1         ST   \n",
       "\n",
       "     MaxHR ExerciseAngina  Oldpeak ST_Slope  HeartDisease  \n",
       "840    132              N      0.0     Flat             0  \n",
       "681    186              Y      0.0       Up             0  \n",
       "395    150              N      0.0     Flat             1  \n",
       "129    152              Y      1.5     Flat             0  \n",
       "793    163              N      0.2     Flat             1  \n",
       "569    110              Y      2.5     Flat             1  \n",
       "831    172              N      0.0       Up             0  \n",
       "650    150              Y      0.0       Up             1  \n",
       "375    121              N      0.0       Up             1  \n",
       "331     82              N     -1.0       Up             1  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.sample(10, random_state=11)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "e70cc5fa",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:43.462722Z",
     "iopub.status.busy": "2024-03-10T10:04:43.462296Z",
     "iopub.status.idle": "2024-03-10T10:04:43.505449Z",
     "shell.execute_reply": "2024-03-10T10:04:43.504386Z"
    },
    "papermill": {
     "duration": 0.084309,
     "end_time": "2024-03-10T10:04:43.508170",
     "exception": false,
     "start_time": "2024-03-10T10:04:43.423861",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>mean</th>\n",
       "      <th>std</th>\n",
       "      <th>min</th>\n",
       "      <th>25%</th>\n",
       "      <th>50%</th>\n",
       "      <th>75%</th>\n",
       "      <th>max</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Age</th>\n",
       "      <td>918.0</td>\n",
       "      <td>53.510893</td>\n",
       "      <td>9.432617</td>\n",
       "      <td>28.0</td>\n",
       "      <td>47.00</td>\n",
       "      <td>54.0</td>\n",
       "      <td>60.0</td>\n",
       "      <td>77.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>RestingBP</th>\n",
       "      <td>918.0</td>\n",
       "      <td>132.396514</td>\n",
       "      <td>18.514154</td>\n",
       "      <td>0.0</td>\n",
       "      <td>120.00</td>\n",
       "      <td>130.0</td>\n",
       "      <td>140.0</td>\n",
       "      <td>200.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Cholesterol</th>\n",
       "      <td>918.0</td>\n",
       "      <td>198.799564</td>\n",
       "      <td>109.384145</td>\n",
       "      <td>0.0</td>\n",
       "      <td>173.25</td>\n",
       "      <td>223.0</td>\n",
       "      <td>267.0</td>\n",
       "      <td>603.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>FastingBS</th>\n",
       "      <td>918.0</td>\n",
       "      <td>0.233115</td>\n",
       "      <td>0.423046</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>MaxHR</th>\n",
       "      <td>918.0</td>\n",
       "      <td>136.809368</td>\n",
       "      <td>25.460334</td>\n",
       "      <td>60.0</td>\n",
       "      <td>120.00</td>\n",
       "      <td>138.0</td>\n",
       "      <td>156.0</td>\n",
       "      <td>202.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Oldpeak</th>\n",
       "      <td>918.0</td>\n",
       "      <td>0.887364</td>\n",
       "      <td>1.066570</td>\n",
       "      <td>-2.6</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.6</td>\n",
       "      <td>1.5</td>\n",
       "      <td>6.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>HeartDisease</th>\n",
       "      <td>918.0</td>\n",
       "      <td>0.553377</td>\n",
       "      <td>0.497414</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              count        mean         std   min     25%    50%    75%    max\n",
       "Age           918.0   53.510893    9.432617  28.0   47.00   54.0   60.0   77.0\n",
       "RestingBP     918.0  132.396514   18.514154   0.0  120.00  130.0  140.0  200.0\n",
       "Cholesterol   918.0  198.799564  109.384145   0.0  173.25  223.0  267.0  603.0\n",
       "FastingBS     918.0    0.233115    0.423046   0.0    0.00    0.0    0.0    1.0\n",
       "MaxHR         918.0  136.809368   25.460334  60.0  120.00  138.0  156.0  202.0\n",
       "Oldpeak       918.0    0.887364    1.066570  -2.6    0.00    0.6    1.5    6.2\n",
       "HeartDisease  918.0    0.553377    0.497414   0.0    0.00    1.0    1.0    1.0"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Summary of All Numerical Data\n",
    "df.describe().T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "ea57591d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:43.582590Z",
     "iopub.status.busy": "2024-03-10T10:04:43.582144Z",
     "iopub.status.idle": "2024-03-10T10:04:43.605773Z",
     "shell.execute_reply": "2024-03-10T10:04:43.604604Z"
    },
    "papermill": {
     "duration": 0.066843,
     "end_time": "2024-03-10T10:04:43.611373",
     "exception": false,
     "start_time": "2024-03-10T10:04:43.544530",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>unique</th>\n",
       "      <th>top</th>\n",
       "      <th>freq</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Sex</th>\n",
       "      <td>918</td>\n",
       "      <td>2</td>\n",
       "      <td>M</td>\n",
       "      <td>725</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ChestPainType</th>\n",
       "      <td>918</td>\n",
       "      <td>4</td>\n",
       "      <td>ASY</td>\n",
       "      <td>496</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>RestingECG</th>\n",
       "      <td>918</td>\n",
       "      <td>3</td>\n",
       "      <td>Normal</td>\n",
       "      <td>552</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ExerciseAngina</th>\n",
       "      <td>918</td>\n",
       "      <td>2</td>\n",
       "      <td>N</td>\n",
       "      <td>547</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ST_Slope</th>\n",
       "      <td>918</td>\n",
       "      <td>3</td>\n",
       "      <td>Flat</td>\n",
       "      <td>460</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               count unique     top freq\n",
       "Sex              918      2       M  725\n",
       "ChestPainType    918      4     ASY  496\n",
       "RestingECG       918      3  Normal  552\n",
       "ExerciseAngina   918      2       N  547\n",
       "ST_Slope         918      3    Flat  460"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Summary of All Categorical Data\n",
    "df.describe(include=\"object\").T"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b630766f",
   "metadata": {
    "papermill": {
     "duration": 0.037999,
     "end_time": "2024-03-10T10:04:43.688274",
     "exception": false,
     "start_time": "2024-03-10T10:04:43.650275",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Lets Check of Null Values & Duplicates Records!!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "5d69f3ee",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:43.765052Z",
     "iopub.status.busy": "2024-03-10T10:04:43.763526Z",
     "iopub.status.idle": "2024-03-10T10:04:43.775538Z",
     "shell.execute_reply": "2024-03-10T10:04:43.774635Z"
    },
    "papermill": {
     "duration": 0.053109,
     "end_time": "2024-03-10T10:04:43.778117",
     "exception": false,
     "start_time": "2024-03-10T10:04:43.725008",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Age               0.0\n",
       "Sex               0.0\n",
       "ChestPainType     0.0\n",
       "RestingBP         0.0\n",
       "Cholesterol       0.0\n",
       "FastingBS         0.0\n",
       "RestingECG        0.0\n",
       "MaxHR             0.0\n",
       "ExerciseAngina    0.0\n",
       "Oldpeak           0.0\n",
       "ST_Slope          0.0\n",
       "HeartDisease      0.0\n",
       "dtype: float64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check Nan Values\n",
    "np.round(df.isna().sum() / len(df) * 100, 3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "15299603",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:43.854319Z",
     "iopub.status.busy": "2024-03-10T10:04:43.853599Z",
     "iopub.status.idle": "2024-03-10T10:04:43.862267Z",
     "shell.execute_reply": "2024-03-10T10:04:43.861414Z"
    },
    "papermill": {
     "duration": 0.04982,
     "end_time": "2024-03-10T10:04:43.864570",
     "exception": false,
     "start_time": "2024-03-10T10:04:43.814750",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check Duplicates Records\n",
    "df.duplicated().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8c54ed49",
   "metadata": {
    "papermill": {
     "duration": 0.036555,
     "end_time": "2024-03-10T10:04:43.938272",
     "exception": false,
     "start_time": "2024-03-10T10:04:43.901717",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<p style = \"color: #F9F07A;\n",
    "            font: bold 24px arial;\n",
    "            background-color: #111;\n",
    "            padding: 15px;\n",
    "            border: 3px solid #FFF67E;\n",
    "            border-radius: 8px\"> \n",
    "    Perfecto 🤗👌,\n",
    "    Dataset almost clean data\n",
    "</p>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4f305883",
   "metadata": {
    "papermill": {
     "duration": 0.038325,
     "end_time": "2024-03-10T10:04:44.014594",
     "exception": false,
     "start_time": "2024-03-10T10:04:43.976269",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Now, Let's Dive Deeper Into Each Column🥽"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "ccc312ac",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:44.093178Z",
     "iopub.status.busy": "2024-03-10T10:04:44.092764Z",
     "iopub.status.idle": "2024-03-10T10:04:44.108413Z",
     "shell.execute_reply": "2024-03-10T10:04:44.107339Z"
    },
    "papermill": {
     "duration": 0.057903,
     "end_time": "2024-03-10T10:04:44.111116",
     "exception": false,
     "start_time": "2024-03-10T10:04:44.053213",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Sex</th>\n",
       "      <th>ChestPainType</th>\n",
       "      <th>RestingBP</th>\n",
       "      <th>Cholesterol</th>\n",
       "      <th>FastingBS</th>\n",
       "      <th>RestingECG</th>\n",
       "      <th>MaxHR</th>\n",
       "      <th>ExerciseAngina</th>\n",
       "      <th>Oldpeak</th>\n",
       "      <th>ST_Slope</th>\n",
       "      <th>HeartDisease</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>40</td>\n",
       "      <td>M</td>\n",
       "      <td>ATA</td>\n",
       "      <td>140</td>\n",
       "      <td>289</td>\n",
       "      <td>0</td>\n",
       "      <td>Normal</td>\n",
       "      <td>172</td>\n",
       "      <td>N</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Up</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>49</td>\n",
       "      <td>F</td>\n",
       "      <td>NAP</td>\n",
       "      <td>160</td>\n",
       "      <td>180</td>\n",
       "      <td>0</td>\n",
       "      <td>Normal</td>\n",
       "      <td>156</td>\n",
       "      <td>N</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Flat</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>37</td>\n",
       "      <td>M</td>\n",
       "      <td>ATA</td>\n",
       "      <td>130</td>\n",
       "      <td>283</td>\n",
       "      <td>0</td>\n",
       "      <td>ST</td>\n",
       "      <td>98</td>\n",
       "      <td>N</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Up</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>48</td>\n",
       "      <td>F</td>\n",
       "      <td>ASY</td>\n",
       "      <td>138</td>\n",
       "      <td>214</td>\n",
       "      <td>0</td>\n",
       "      <td>Normal</td>\n",
       "      <td>108</td>\n",
       "      <td>Y</td>\n",
       "      <td>1.5</td>\n",
       "      <td>Flat</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>54</td>\n",
       "      <td>M</td>\n",
       "      <td>NAP</td>\n",
       "      <td>150</td>\n",
       "      <td>195</td>\n",
       "      <td>0</td>\n",
       "      <td>Normal</td>\n",
       "      <td>122</td>\n",
       "      <td>N</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Up</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Age Sex ChestPainType  RestingBP  Cholesterol  FastingBS RestingECG  MaxHR  \\\n",
       "0   40   M           ATA        140          289          0     Normal    172   \n",
       "1   49   F           NAP        160          180          0     Normal    156   \n",
       "2   37   M           ATA        130          283          0         ST     98   \n",
       "3   48   F           ASY        138          214          0     Normal    108   \n",
       "4   54   M           NAP        150          195          0     Normal    122   \n",
       "\n",
       "  ExerciseAngina  Oldpeak ST_Slope  HeartDisease  \n",
       "0              N      0.0       Up             0  \n",
       "1              N      1.0     Flat             1  \n",
       "2              N      0.0       Up             0  \n",
       "3              Y      1.5     Flat             1  \n",
       "4              N      0.0       Up             0  "
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "504c407f",
   "metadata": {
    "papermill": {
     "duration": 0.036926,
     "end_time": "2024-03-10T10:04:44.185382",
     "exception": false,
     "start_time": "2024-03-10T10:04:44.148456",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<h2 style=\"font: bold 26px tahoma\">\n",
    "    ♠ Age Column 👨👨‍🦳\n",
    "</h2>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "008ca385",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:44.264147Z",
     "iopub.status.busy": "2024-03-10T10:04:44.263702Z",
     "iopub.status.idle": "2024-03-10T10:04:44.270582Z",
     "shell.execute_reply": "2024-03-10T10:04:44.269161Z"
    },
    "papermill": {
     "duration": 0.049807,
     "end_time": "2024-03-10T10:04:44.273228",
     "exception": false,
     "start_time": "2024-03-10T10:04:44.223421",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "mean_of_age = df[\"Age\"].mean()\n",
    "median_of_age = df[\"Age\"].median()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "8f6d614f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:44.350740Z",
     "iopub.status.busy": "2024-03-10T10:04:44.349982Z",
     "iopub.status.idle": "2024-03-10T10:04:46.622785Z",
     "shell.execute_reply": "2024-03-10T10:04:46.621249Z"
    },
    "papermill": {
     "duration": 2.314909,
     "end_time": "2024-03-10T10:04:46.625678",
     "exception": false,
     "start_time": "2024-03-10T10:04:44.310769",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "        <script type=\"text/javascript\">\n",
       "        window.PlotlyConfig = {MathJaxConfig: 'local'};\n",
       "        if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: \"STIX-Web\"}});}\n",
       "        if (typeof require !== 'undefined') {\n",
       "        require.undef(\"plotly\");\n",
       "        requirejs.config({\n",
       "            paths: {\n",
       "                'plotly': ['https://cdn.plot.ly/plotly-2.27.0.min']\n",
       "            }\n",
       "        });\n",
       "        require(['plotly'], function(Plotly) {\n",
       "            window._Plotly = Plotly;\n",
       "        });\n",
       "        }\n",
       "        </script>\n",
       "        "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<div>                            <div id=\"3aed3eb7-6d72-4e9d-b655-fdef6da1ab44\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"3aed3eb7-6d72-4e9d-b655-fdef6da1ab44\")) {                    Plotly.newPlot(                        \"3aed3eb7-6d72-4e9d-b655-fdef6da1ab44\",                        [{\"alignmentgroup\":\"True\",\"hovertemplate\":\"Age=%{x}\\u003cextra\\u003e\\u003c\\u002fextra\\u003e\",\"legendgroup\":\"\",\"marker\":{\"color\":\"#636efa\"},\"name\":\"\",\"notched\":false,\"offsetgroup\":\"\",\"orientation\":\"h\",\"showlegend\":false,\"x\":[40,49,37,48,54,39,45,54,37,48,37,58,39,49,42,54,38,43,60,36,43,44,49,44,40,36,53,52,53,51,53,56,54,41,43,32,65,41,48,48,54,54,35,52,43,59,37,50,36,41,50,47,45,41,52,51,31,58,54,52,49,43,45,46,50,37,45,32,52,44,57,44,52,44,55,46,32,35,52,49,55,54,63,52,56,66,65,53,43,55,49,39,52,48,39,58,43,39,56,41,65,51,40,40,46,57,48,34,50,39,59,57,47,38,49,33,38,59,35,34,47,52,46,58,58,54,34,48,54,42,38,46,56,56,61,49,43,39,54,43,52,50,47,53,56,39,42,43,50,54,39,48,40,55,41,56,38,49,44,54,59,49,47,42,52,46,50,48,58,58,29,40,53,49,52,43,54,59,37,46,52,51,52,46,54,58,58,41,50,53,46,50,48,45,41,62,49,42,53,57,47,46,42,31,56,50,35,35,28,54,48,50,56,56,47,30,39,54,55,29,46,51,48,33,55,50,53,38,41,37,37,40,38,41,54,39,41,55,48,48,55,54,55,43,48,54,54,48,45,49,44,48,61,62,55,53,55,36,51,55,46,54,46,59,47,54,52,34,54,47,45,32,55,55,45,59,51,52,57,54,60,49,51,55,42,51,59,53,48,36,48,47,53,65,32,61,50,57,51,47,60,55,53,62,51,51,55,53,58,57,65,60,41,34,53,74,57,56,61,68,59,63,38,62,46,42,45,59,52,60,60,56,38,40,51,62,72,63,63,64,43,64,61,52,51,69,59,48,69,36,53,43,56,58,55,67,46,53,38,53,62,47,56,56,56,64,61,68,57,63,60,66,63,59,61,73,47,65,70,50,60,50,43,38,54,61,42,53,55,61,51,70,61,38,57,38,62,58,52,61,50,51,65,52,47,35,57,62,59,53,62,54,56,56,54,66,63,44,60,55,66,66,65,60,60,60,56,59,62,63,57,62,63,46,63,60,58,64,63,74,52,69,51,60,56,55,54,77,63,55,52,64,60,60,58,59,61,40,61,41,57,63,59,51,59,42,55,63,62,56,53,68,53,60,62,59,51,61,57,56,58,69,67,58,65,63,55,57,65,54,72,75,49,51,60,64,58,61,67,62,65,63,69,51,62,55,75,40,67,58,60,63,35,62,43,63,68,65,48,63,64,61,50,59,55,45,65,61,49,72,50,64,55,63,59,56,62,74,54,57,62,76,54,70,61,48,48,61,66,68,55,62,71,74,53,58,75,56,58,64,54,54,59,55,57,61,41,71,38,55,56,69,64,72,69,56,62,67,57,69,51,48,69,69,64,57,53,37,67,74,63,58,61,64,58,60,57,55,55,56,57,61,61,74,68,51,62,53,62,46,54,62,55,58,62,70,67,57,64,74,65,56,59,60,63,59,53,44,61,57,71,46,53,64,40,67,48,43,47,54,48,46,51,58,71,57,66,37,59,50,48,61,59,42,48,40,62,44,46,59,58,49,44,66,65,42,52,65,63,45,41,61,60,59,62,57,51,44,60,63,57,51,58,44,47,61,57,70,76,67,45,45,39,42,56,58,35,58,41,57,42,62,59,41,50,59,61,54,54,52,47,66,58,64,50,44,67,49,57,63,48,51,60,59,45,55,41,60,54,42,49,46,56,66,56,49,54,57,65,54,54,62,52,52,60,63,66,42,64,54,46,67,56,34,57,64,59,50,51,54,53,52,40,58,41,41,50,54,64,51,46,55,45,56,66,38,62,55,58,43,64,50,53,45,65,69,69,67,68,34,62,51,46,67,50,42,56,41,42,53,43,56,52,62,70,54,70,54,35,48,55,58,54,69,77,68,58,60,51,55,52,60,58,64,37,59,51,43,58,29,41,63,51,54,44,54,65,57,63,35,41,62,43,58,52,61,39,45,52,62,62,53,43,47,52,68,39,53,62,51,60,65,65,60,60,54,44,44,51,59,71,61,55,64,43,58,60,58,49,48,52,44,56,57,67,53,52,43,52,59,64,66,39,57,58,57,47,55,35,61,58,58,58,56,56,67,55,44,63,63,41,59,57,45,68,57,57,38],\"x0\":\" \",\"xaxis\":\"x\",\"y0\":\" \",\"yaxis\":\"y\",\"type\":\"box\"}],                        {\"boxmode\":\"group\",\"hoverlabel\":{\"bgcolor\":\"#111\",\"font\":{\"family\":\"arial\",\"size\":16}},\"legend\":{\"tracegroupgap\":0},\"showlegend\":false,\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"white\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"#C8D4E3\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"white\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"radialaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"yaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"zaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"caxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2}}},\"title\":{\"font\":{\"family\":\"tahoma\",\"size\":28},\"text\":\"Ages 5-Number Summary\"},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Age\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0]}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('3aed3eb7-6d72-4e9d-b655-fdef6da1ab44');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = px.box(\n",
    "    x=df[\"Age\"], \n",
    "    title= \"Ages 5-Number Summary\",\n",
    "    template=\"plotly_white\",\n",
    "    labels={\"x\" :\"Age\"},\n",
    ")\n",
    "custome_layout()\n",
    "\n",
    "iplot(fig)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "03e47a83",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:46.703730Z",
     "iopub.status.busy": "2024-03-10T10:04:46.703236Z",
     "iopub.status.idle": "2024-03-10T10:04:47.111016Z",
     "shell.execute_reply": "2024-03-10T10:04:47.109565Z"
    },
    "papermill": {
     "duration": 0.450023,
     "end_time": "2024-03-10T10:04:47.113804",
     "exception": false,
     "start_time": "2024-03-10T10:04:46.663781",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>                            <div id=\"8d0ecec1-8e15-4d98-bb94-96a8df2526b3\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"8d0ecec1-8e15-4d98-bb94-96a8df2526b3\")) {                    Plotly.newPlot(                        \"8d0ecec1-8e15-4d98-bb94-96a8df2526b3\",                        [{\"alignmentgroup\":\"True\",\"bingroup\":\"x\",\"hovertemplate\":\"Age: %{x}\\u003cbr\\u003eFrequency: %{y}\",\"legendgroup\":\"Age\",\"marker\":{\"color\":\"#636efa\",\"line\":{\"color\":\"#000\",\"width\":0.2},\"pattern\":{\"shape\":\"\"}},\"name\":\"Age\",\"nbinsx\":25,\"offsetgroup\":\"Age\",\"orientation\":\"v\",\"showlegend\":true,\"textfont\":{\"color\":\"#fff\",\"family\":\"tahoma\",\"size\":20},\"x\":[40,49,37,48,54,39,45,54,37,48,37,58,39,49,42,54,38,43,60,36,43,44,49,44,40,36,53,52,53,51,53,56,54,41,43,32,65,41,48,48,54,54,35,52,43,59,37,50,36,41,50,47,45,41,52,51,31,58,54,52,49,43,45,46,50,37,45,32,52,44,57,44,52,44,55,46,32,35,52,49,55,54,63,52,56,66,65,53,43,55,49,39,52,48,39,58,43,39,56,41,65,51,40,40,46,57,48,34,50,39,59,57,47,38,49,33,38,59,35,34,47,52,46,58,58,54,34,48,54,42,38,46,56,56,61,49,43,39,54,43,52,50,47,53,56,39,42,43,50,54,39,48,40,55,41,56,38,49,44,54,59,49,47,42,52,46,50,48,58,58,29,40,53,49,52,43,54,59,37,46,52,51,52,46,54,58,58,41,50,53,46,50,48,45,41,62,49,42,53,57,47,46,42,31,56,50,35,35,28,54,48,50,56,56,47,30,39,54,55,29,46,51,48,33,55,50,53,38,41,37,37,40,38,41,54,39,41,55,48,48,55,54,55,43,48,54,54,48,45,49,44,48,61,62,55,53,55,36,51,55,46,54,46,59,47,54,52,34,54,47,45,32,55,55,45,59,51,52,57,54,60,49,51,55,42,51,59,53,48,36,48,47,53,65,32,61,50,57,51,47,60,55,53,62,51,51,55,53,58,57,65,60,41,34,53,74,57,56,61,68,59,63,38,62,46,42,45,59,52,60,60,56,38,40,51,62,72,63,63,64,43,64,61,52,51,69,59,48,69,36,53,43,56,58,55,67,46,53,38,53,62,47,56,56,56,64,61,68,57,63,60,66,63,59,61,73,47,65,70,50,60,50,43,38,54,61,42,53,55,61,51,70,61,38,57,38,62,58,52,61,50,51,65,52,47,35,57,62,59,53,62,54,56,56,54,66,63,44,60,55,66,66,65,60,60,60,56,59,62,63,57,62,63,46,63,60,58,64,63,74,52,69,51,60,56,55,54,77,63,55,52,64,60,60,58,59,61,40,61,41,57,63,59,51,59,42,55,63,62,56,53,68,53,60,62,59,51,61,57,56,58,69,67,58,65,63,55,57,65,54,72,75,49,51,60,64,58,61,67,62,65,63,69,51,62,55,75,40,67,58,60,63,35,62,43,63,68,65,48,63,64,61,50,59,55,45,65,61,49,72,50,64,55,63,59,56,62,74,54,57,62,76,54,70,61,48,48,61,66,68,55,62,71,74,53,58,75,56,58,64,54,54,59,55,57,61,41,71,38,55,56,69,64,72,69,56,62,67,57,69,51,48,69,69,64,57,53,37,67,74,63,58,61,64,58,60,57,55,55,56,57,61,61,74,68,51,62,53,62,46,54,62,55,58,62,70,67,57,64,74,65,56,59,60,63,59,53,44,61,57,71,46,53,64,40,67,48,43,47,54,48,46,51,58,71,57,66,37,59,50,48,61,59,42,48,40,62,44,46,59,58,49,44,66,65,42,52,65,63,45,41,61,60,59,62,57,51,44,60,63,57,51,58,44,47,61,57,70,76,67,45,45,39,42,56,58,35,58,41,57,42,62,59,41,50,59,61,54,54,52,47,66,58,64,50,44,67,49,57,63,48,51,60,59,45,55,41,60,54,42,49,46,56,66,56,49,54,57,65,54,54,62,52,52,60,63,66,42,64,54,46,67,56,34,57,64,59,50,51,54,53,52,40,58,41,41,50,54,64,51,46,55,45,56,66,38,62,55,58,43,64,50,53,45,65,69,69,67,68,34,62,51,46,67,50,42,56,41,42,53,43,56,52,62,70,54,70,54,35,48,55,58,54,69,77,68,58,60,51,55,52,60,58,64,37,59,51,43,58,29,41,63,51,54,44,54,65,57,63,35,41,62,43,58,52,61,39,45,52,62,62,53,43,47,52,68,39,53,62,51,60,65,65,60,60,54,44,44,51,59,71,61,55,64,43,58,60,58,49,48,52,44,56,57,67,53,52,43,52,59,64,66,39,57,58,57,47,55,35,61,58,58,58,56,56,67,55,44,63,63,41,59,57,45,68,57,57,38],\"xaxis\":\"x\",\"yaxis\":\"y\",\"type\":\"histogram\"}],                        {\"barmode\":\"relative\",\"hoverlabel\":{\"bgcolor\":\"#111\",\"font\":{\"family\":\"arial\",\"size\":16}},\"legend\":{\"title\":{\"text\":\"variable\"},\"tracegroupgap\":0},\"shapes\":[{\"label\":{\"font\":{\"color\":\"#AF2655\",\"family\":\"arial\",\"size\":14},\"text\":\"     Mean:  53.5    \",\"textangle\":0,\"textposition\":\"end\",\"xanchor\":\"right\",\"yanchor\":\"top\"},\"line\":{\"color\":\"#AF2655\",\"dash\":\"dashdot\",\"width\":3},\"type\":\"line\",\"x0\":53.510893246187365,\"x1\":53.510893246187365,\"y0\":0,\"y1\":97},{\"label\":{\"font\":{\"color\":\"#0C356A\",\"family\":\"arial\",\"size\":14},\"text\":\"     Median:  54.0  \",\"textangle\":0,\"textposition\":\"end\",\"xanchor\":\"left\",\"yanchor\":\"top\"},\"line\":{\"color\":\"#0C356A\",\"dash\":\"dashdot\",\"width\":3},\"type\":\"line\",\"x0\":54.0,\"x1\":54.0,\"y0\":0,\"y1\":97}],\"showlegend\":false,\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"white\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"#C8D4E3\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"white\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"radialaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"yaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"zaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"caxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2}}},\"title\":{\"font\":{\"family\":\"tahoma\",\"size\":28},\"text\":\"Age Distribution\"},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Age\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"count\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('8d0ecec1-8e15-4d98-bb94-96a8df2526b3');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = px.histogram(\n",
    "    df[\"Age\"], \n",
    "    nbins=25,\n",
    "    title= \"Age Distribution\",\n",
    "    template=\"plotly_white\",\n",
    "    labels={\"value\" :\"Age\"}\n",
    ")\n",
    "\n",
    "custome_layout()\n",
    "fig.update_traces(\n",
    "    textfont = {\n",
    "        \"size\" : 20,\n",
    "        \"family\" :\"tahoma\",\n",
    "        \"color\": \"#fff\"\n",
    "    },\n",
    "    hovertemplate = \"Age: %{x}<br>Frequency: %{y}\",\n",
    "    marker=dict(line=dict(color='#000', width=0.2))\n",
    ")\n",
    "\n",
    "\n",
    "# Adding Mean Line\n",
    "add_line(x0=mean_of_age, y0=0, x1=mean_of_age, y1=95, line_color=\"#AF2655\",font_color=\"#AF2655\", \n",
    "         text=\"Mean\", xposition=\"right\")\n",
    "\n",
    "# Adding Median Line\n",
    "add_line(x0=median_of_age, y0=0, x1=median_of_age, y1=95, line_color=\"#0C356A\",\n",
    "         font_color=\"#0C356A\", xposition=\"left\", text=\"Median\")\n",
    "\n",
    "iplot(fig)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "03046cd4",
   "metadata": {
    "papermill": {
     "duration": 0.037483,
     "end_time": "2024-03-10T10:04:47.188719",
     "exception": false,
     "start_time": "2024-03-10T10:04:47.151236",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<p style = \"color: #F9F07A;\n",
    "            font: bold 20px arial;\n",
    "            padding: 15px;\n",
    "            background-color: #111\">\n",
    " ♦ From Histogram & Box Plot🤔\n",
    "    <br>\n",
    "    🤗 We Can Say, The Age has a Slight Left Skewness (Considered Approximately Symmetric)\n",
    "</p>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0492115f",
   "metadata": {
    "papermill": {
     "duration": 0.036838,
     "end_time": "2024-03-10T10:04:47.262544",
     "exception": false,
     "start_time": "2024-03-10T10:04:47.225706",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<h2 style=\"font: bold 26px tahoma\">\n",
    "    ♠ Gender Column 👨👩\n",
    "</h2>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "fdb1ca75",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:47.339952Z",
     "iopub.status.busy": "2024-03-10T10:04:47.339057Z",
     "iopub.status.idle": "2024-03-10T10:04:47.350132Z",
     "shell.execute_reply": "2024-03-10T10:04:47.348782Z"
    },
    "papermill": {
     "duration": 0.053475,
     "end_time": "2024-03-10T10:04:47.352830",
     "exception": false,
     "start_time": "2024-03-10T10:04:47.299355",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Sex\n",
       "M    78.98%\n",
       "F    21.02%\n",
       "Name: proportion, dtype: object"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "gender = df[\"Sex\"].value_counts(normalize=1) * 100\n",
    "gender.apply(lambda x: f\"{x:0.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "c33dbfdd",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:47.433201Z",
     "iopub.status.busy": "2024-03-10T10:04:47.431833Z",
     "iopub.status.idle": "2024-03-10T10:04:47.598133Z",
     "shell.execute_reply": "2024-03-10T10:04:47.596809Z"
    },
    "papermill": {
     "duration": 0.209041,
     "end_time": "2024-03-10T10:04:47.600906",
     "exception": false,
     "start_time": "2024-03-10T10:04:47.391865",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>                            <div id=\"3a0e1f6b-c47c-4d1b-8b67-f148dd889209\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"3a0e1f6b-c47c-4d1b-8b67-f148dd889209\")) {                    Plotly.newPlot(                        \"3a0e1f6b-c47c-4d1b-8b67-f148dd889209\",                        [{\"alignmentgroup\":\"True\",\"hovertemplate\":\"Gender: %{x}\\u003cbr\\u003ePercentage: %{y:0.2f}%\",\"legendgroup\":\"M\",\"marker\":{\"color\":\"#636efa\",\"pattern\":{\"shape\":\"\"}},\"name\":\"M\",\"offsetgroup\":\"M\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"79%\"],\"textfont\":{\"color\":\"#fff\",\"family\":\"consolas\",\"size\":18},\"textposition\":\"auto\",\"x\":[\"Male\"],\"xaxis\":\"x\",\"y\":[78.9760348583878],\"yaxis\":\"y\",\"type\":\"bar\"},{\"alignmentgroup\":\"True\",\"hovertemplate\":\"Gender: %{x}\\u003cbr\\u003ePercentage: %{y:0.2f}%\",\"legendgroup\":\"F\",\"marker\":{\"color\":\"#EF553B\",\"pattern\":{\"shape\":\"\"}},\"name\":\"F\",\"offsetgroup\":\"F\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"21%\"],\"textfont\":{\"color\":\"#fff\",\"family\":\"consolas\",\"size\":18},\"textposition\":\"auto\",\"x\":[\"Female\"],\"xaxis\":\"x\",\"y\":[21.0239651416122],\"yaxis\":\"y\",\"type\":\"bar\"}],                        {\"barmode\":\"relative\",\"hoverlabel\":{\"bgcolor\":\"#111\",\"font\":{\"family\":\"arial\",\"size\":16}},\"legend\":{\"title\":{\"text\":\"color\"},\"tracegroupgap\":0},\"showlegend\":false,\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"white\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"#C8D4E3\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"white\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"radialaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"yaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"zaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"caxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2}}},\"title\":{\"font\":{\"family\":\"tahoma\",\"size\":28},\"text\":\"Percentage of Gender Frequency👨👩\"},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Gender\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Frequency in PCT(%)\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('3a0e1f6b-c47c-4d1b-8b67-f148dd889209');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = count_bar_plot(\n",
    "    df, \"Sex\",x_title= \"Gender\", y_title=\"Frequency in PCT(%)\", \n",
    "    title=\"Percentage of Gender Frequency👨👩\",\n",
    "    hover_template=\"Gender: %{x}<br>Percentage: %{y:0.2f}%\",\n",
    "    bars_names = [\"Male\" if i == \"M\" else \"Female\" for i in gender.index]        \n",
    ")\n",
    "\n",
    "iplot(fig)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "85b80ed9",
   "metadata": {
    "papermill": {
     "duration": 0.042007,
     "end_time": "2024-03-10T10:04:47.682604",
     "exception": false,
     "start_time": "2024-03-10T10:04:47.640597",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<h2 style=\"font: bold 26px tahoma\">\n",
    "    ♠ Chest Pain Type Column \n",
    "</h2>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "887e22d8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:47.765488Z",
     "iopub.status.busy": "2024-03-10T10:04:47.765055Z",
     "iopub.status.idle": "2024-03-10T10:04:47.776226Z",
     "shell.execute_reply": "2024-03-10T10:04:47.775308Z"
    },
    "papermill": {
     "duration": 0.054697,
     "end_time": "2024-03-10T10:04:47.778479",
     "exception": false,
     "start_time": "2024-03-10T10:04:47.723782",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ChestPainType\n",
       "ASY    54.03%\n",
       "NAP    22.11%\n",
       "ATA    18.85%\n",
       "TA      5.01%\n",
       "Name: proportion, dtype: object"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# ===== To Know each of Thees Abbreviation Go To The Columns Details 💊 Section =====\n",
    "ChestPainType = df[\"ChestPainType\"].value_counts(normalize=1) * 100\n",
    "ChestPainType.apply(lambda x: f\"{x:0.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "ce4aa82a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:47.860176Z",
     "iopub.status.busy": "2024-03-10T10:04:47.859721Z",
     "iopub.status.idle": "2024-03-10T10:04:47.995434Z",
     "shell.execute_reply": "2024-03-10T10:04:47.994130Z"
    },
    "papermill": {
     "duration": 0.179959,
     "end_time": "2024-03-10T10:04:47.998473",
     "exception": false,
     "start_time": "2024-03-10T10:04:47.818514",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>                            <div id=\"945a4e19-d85a-4bda-8edb-8137acf2017b\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"945a4e19-d85a-4bda-8edb-8137acf2017b\")) {                    Plotly.newPlot(                        \"945a4e19-d85a-4bda-8edb-8137acf2017b\",                        [{\"alignmentgroup\":\"True\",\"hovertemplate\":\"Chest Pain Type: %{x}\\u003cbr\\u003ePercentage: %{y:0.2f}%\",\"legendgroup\":\"ASY\",\"marker\":{\"color\":\"#636efa\",\"pattern\":{\"shape\":\"\"}},\"name\":\"ASY\",\"offsetgroup\":\"ASY\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"54%\"],\"textfont\":{\"color\":\"#fff\",\"family\":\"consolas\",\"size\":18},\"textposition\":\"auto\",\"x\":[\"ASY\"],\"xaxis\":\"x\",\"y\":[54.03050108932461],\"yaxis\":\"y\",\"type\":\"bar\"},{\"alignmentgroup\":\"True\",\"hovertemplate\":\"Chest Pain Type: %{x}\\u003cbr\\u003ePercentage: %{y:0.2f}%\",\"legendgroup\":\"NAP\",\"marker\":{\"color\":\"#EF553B\",\"pattern\":{\"shape\":\"\"}},\"name\":\"NAP\",\"offsetgroup\":\"NAP\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"22%\"],\"textfont\":{\"color\":\"#fff\",\"family\":\"consolas\",\"size\":18},\"textposition\":\"auto\",\"x\":[\"NAP\"],\"xaxis\":\"x\",\"y\":[22.11328976034858],\"yaxis\":\"y\",\"type\":\"bar\"},{\"alignmentgroup\":\"True\",\"hovertemplate\":\"Chest Pain Type: %{x}\\u003cbr\\u003ePercentage: %{y:0.2f}%\",\"legendgroup\":\"ATA\",\"marker\":{\"color\":\"#00cc96\",\"pattern\":{\"shape\":\"\"}},\"name\":\"ATA\",\"offsetgroup\":\"ATA\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"19%\"],\"textfont\":{\"color\":\"#fff\",\"family\":\"consolas\",\"size\":18},\"textposition\":\"auto\",\"x\":[\"ATA\"],\"xaxis\":\"x\",\"y\":[18.845315904139433],\"yaxis\":\"y\",\"type\":\"bar\"},{\"alignmentgroup\":\"True\",\"hovertemplate\":\"Chest Pain Type: %{x}\\u003cbr\\u003ePercentage: %{y:0.2f}%\",\"legendgroup\":\"TA\",\"marker\":{\"color\":\"#ab63fa\",\"pattern\":{\"shape\":\"\"}},\"name\":\"TA\",\"offsetgroup\":\"TA\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"5%\"],\"textfont\":{\"color\":\"#fff\",\"family\":\"consolas\",\"size\":18},\"textposition\":\"auto\",\"x\":[\"TA\"],\"xaxis\":\"x\",\"y\":[5.010893246187363],\"yaxis\":\"y\",\"type\":\"bar\"}],                        {\"barmode\":\"relative\",\"hoverlabel\":{\"bgcolor\":\"#111\",\"font\":{\"family\":\"arial\",\"size\":16}},\"legend\":{\"title\":{\"text\":\"color\"},\"tracegroupgap\":0},\"showlegend\":false,\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"white\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"#C8D4E3\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"white\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"radialaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"yaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"zaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"caxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2}}},\"title\":{\"font\":{\"family\":\"tahoma\",\"size\":28},\"text\":\"Chest Pain Type Frequency\"},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"ChestPainType\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Frequency in PCT(%)\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('945a4e19-d85a-4bda-8edb-8137acf2017b');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = count_bar_plot(\n",
    "    data_frame=df, column_name=\"ChestPainType\", x_title= \"Chest Pain Type\", y_title=\"Frequency in PCT(%)\", \n",
    "    title=\"Chest Pain Type Frequency\",\n",
    "    hover_template=\"Chest Pain Type: %{x}<br>Percentage: %{y:0.2f}%\",\n",
    ")\n",
    "\n",
    "iplot(fig)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eb2a0516",
   "metadata": {
    "papermill": {
     "duration": 0.040868,
     "end_time": "2024-03-10T10:04:48.080130",
     "exception": false,
     "start_time": "2024-03-10T10:04:48.039262",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<h2 style=\"font: bold 26px tahoma\">\n",
    "    ♠ Resting Blood Presur Column 🩸\n",
    "</h2>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "cd374aab",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:48.162153Z",
     "iopub.status.busy": "2024-03-10T10:04:48.161685Z",
     "iopub.status.idle": "2024-03-10T10:04:48.260243Z",
     "shell.execute_reply": "2024-03-10T10:04:48.259290Z"
    },
    "papermill": {
     "duration": 0.142383,
     "end_time": "2024-03-10T10:04:48.262490",
     "exception": false,
     "start_time": "2024-03-10T10:04:48.120107",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>                            <div id=\"a31335ef-41c9-41e5-a00f-e34cf925e6c3\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"a31335ef-41c9-41e5-a00f-e34cf925e6c3\")) {                    Plotly.newPlot(                        \"a31335ef-41c9-41e5-a00f-e34cf925e6c3\",                        [{\"alignmentgroup\":\"True\",\"hovertemplate\":\"Resting Bllod Pressure=%{x}\\u003cextra\\u003e\\u003c\\u002fextra\\u003e\",\"legendgroup\":\"\",\"marker\":{\"color\":\"#636efa\"},\"name\":\"\",\"notched\":false,\"offsetgroup\":\"\",\"orientation\":\"h\",\"showlegend\":false,\"x\":[140,160,130,138,150,120,130,110,140,120,130,136,120,140,115,120,110,120,100,120,100,120,124,150,130,130,124,120,113,125,145,130,125,130,150,125,140,110,120,150,150,130,150,140,120,130,120,140,112,110,130,120,140,130,130,160,120,130,150,112,100,150,140,120,110,120,132,110,160,150,140,130,120,120,140,150,118,140,140,130,110,120,150,160,150,140,170,140,120,140,110,130,120,160,110,130,142,160,120,125,130,130,150,120,118,140,120,150,140,190,130,150,140,140,130,100,120,130,120,140,135,125,110,180,130,120,130,108,120,120,145,110,170,150,130,115,120,120,140,150,160,140,160,140,120,110,120,120,120,130,130,100,130,120,120,155,110,140,130,160,140,128,160,120,140,140,140,140,135,140,120,140,140,140,140,140,140,140,130,130,130,130,140,110,160,160,130,120,120,180,180,170,130,135,125,160,120,150,120,130,110,120,160,100,130,150,120,110,130,125,106,140,130,130,150,170,110,120,140,140,130,160,120,120,120,145,120,92,120,130,130,130,120,112,140,120,120,140,160,160,145,200,160,120,160,120,120,122,130,130,135,120,125,140,145,120,130,150,150,122,140,120,120,130,140,160,130,98,130,130,120,105,140,120,180,180,135,170,180,130,120,150,130,110,140,110,140,120,133,120,110,140,130,115,95,105,145,110,110,110,160,140,125,120,95,120,115,130,115,95,155,125,125,115,80,145,105,140,130,145,125,100,105,115,100,105,110,125,95,130,115,115,100,95,130,120,160,150,140,95,100,110,110,130,120,135,120,115,137,110,120,140,120,130,120,145,115,120,115,105,160,160,155,120,120,200,150,135,140,150,135,150,185,135,125,160,155,160,140,120,160,115,115,110,120,150,145,130,140,160,140,115,130,150,160,135,140,170,165,200,160,130,145,135,110,120,140,115,110,160,150,180,125,125,130,155,140,130,132,142,110,120,150,180,120,160,126,140,110,133,128,120,170,110,126,152,116,120,130,138,128,130,128,130,120,136,130,124,160,0,122,144,140,120,136,154,120,125,134,104,139,136,122,128,131,134,120,132,152,124,126,138,154,141,131,178,132,110,130,170,126,140,142,120,134,139,110,140,140,136,120,170,130,137,142,142,132,146,160,135,136,130,140,132,158,136,136,106,120,110,136,160,123,112,122,130,150,150,102,96,130,120,144,124,150,130,144,139,131,143,133,143,116,110,125,130,133,150,130,110,138,104,138,170,140,132,132,142,112,139,172,120,144,145,155,150,160,137,137,134,133,132,140,135,144,141,150,130,110,158,128,140,150,160,142,137,139,146,156,145,131,140,122,142,141,180,124,118,140,140,136,100,190,130,160,130,122,133,120,130,130,140,120,155,134,114,160,144,158,134,127,135,122,140,120,130,115,124,128,120,120,130,110,140,150,135,142,140,134,128,112,140,140,110,140,120,130,115,112,132,130,138,120,112,110,128,160,120,170,144,130,140,160,130,122,152,124,130,101,126,140,118,110,160,150,136,128,140,140,130,105,138,120,174,120,150,130,120,150,145,150,140,136,118,108,120,120,156,140,106,142,104,94,120,120,146,120,150,130,110,148,128,178,126,150,140,130,124,110,125,110,120,100,140,120,108,120,130,165,130,124,100,150,140,112,180,110,158,135,120,134,120,200,150,130,120,122,152,160,125,160,120,136,134,117,108,112,140,120,150,142,152,125,118,132,145,138,140,125,192,123,112,110,132,112,112,120,108,130,130,105,140,128,120,178,120,150,130,128,110,180,110,130,138,138,160,140,100,120,118,138,140,150,125,129,120,134,110,102,130,130,132,108,140,160,140,145,108,126,124,135,100,110,140,125,118,125,125,140,160,152,102,105,125,130,170,125,122,128,130,130,135,94,120,120,110,135,150,130,138,135,130,132,150,118,145,118,115,128,130,160,138,120,138,120,180,140,130,140,140,130,110,155,140,145,120,130,112,110,150,160,150,132,140,150,120,130,120,130,110,172,120,140,140,160,128,138,132,128,134,170,146,138,154,130,110,130,128,122,148,114,170,125,130,120,152,132,120,140,124,120,164,140,110,144,130,130,138],\"x0\":\" \",\"xaxis\":\"x\",\"y0\":\" \",\"yaxis\":\"y\",\"type\":\"box\"}],                        {\"boxmode\":\"group\",\"hoverlabel\":{\"bgcolor\":\"#111\",\"font\":{\"family\":\"arial\",\"size\":16}},\"legend\":{\"tracegroupgap\":0},\"showlegend\":false,\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"white\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"#C8D4E3\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"white\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"radialaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"yaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"zaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"caxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2}}},\"title\":{\"font\":{\"family\":\"tahoma\",\"size\":28},\"text\":\"Resting Blood Pressure Distribution\"},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Resting Bllod Pressure\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0]}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('a31335ef-41c9-41e5-a00f-e34cf925e6c3');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = px.box(\n",
    "    x = df[\"RestingBP\"], \n",
    "    title= \"Resting Blood Pressure Distribution\",\n",
    "    template=\"plotly_white\",\n",
    "    labels={\"x\": \"Resting Bllod Pressure\"},\n",
    ")\n",
    "\n",
    "custome_layout()\n",
    "\n",
    "iplot(fig)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "80775699",
   "metadata": {
    "papermill": {
     "duration": 0.03993,
     "end_time": "2024-03-10T10:04:48.342314",
     "exception": false,
     "start_time": "2024-03-10T10:04:48.302384",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<p style = \"color: #F9F07A;\n",
    "            font: bold 20px arial;\n",
    "            background-color: #111;\n",
    "            padding: 20px;\n",
    "            border: 3px solid #FFF67E;\n",
    "            border-radius: 8px\"> \n",
    "     ♦ From Box Plot, We Can See The Outliers.\n",
    "</p>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "83c598b6",
   "metadata": {
    "papermill": {
     "duration": 0.039621,
     "end_time": "2024-03-10T10:04:48.421954",
     "exception": false,
     "start_time": "2024-03-10T10:04:48.382333",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<p style = \"color: #F9F07A;\n",
    "            font: bold 18px arial;\n",
    "            background-color: #111;\n",
    "            padding: 20px;\n",
    "            border: 3px solid #FFF67E;\n",
    "            border-radius: 8px\"> \n",
    "    • After searching, i found that, High blood pressure is a risk factor for heart disease.\n",
    "    <br>\n",
    "    <br>\n",
    "    ♠ So, let's check check the heart disease for those who have high blood pressure greater than 170 mmHg (upper)\n",
    "</p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "6679245b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:48.506742Z",
     "iopub.status.busy": "2024-03-10T10:04:48.505960Z",
     "iopub.status.idle": "2024-03-10T10:04:48.625772Z",
     "shell.execute_reply": "2024-03-10T10:04:48.624535Z"
    },
    "papermill": {
     "duration": 0.166437,
     "end_time": "2024-03-10T10:04:48.628444",
     "exception": false,
     "start_time": "2024-03-10T10:04:48.462007",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>                            <div id=\"c4350820-4570-42a8-841f-4e824bfd99e6\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"c4350820-4570-42a8-841f-4e824bfd99e6\")) {                    Plotly.newPlot(                        \"c4350820-4570-42a8-841f-4e824bfd99e6\",                        [{\"alignmentgroup\":\"True\",\"hovertemplate\":\"%{x}\\u003cbr\\u003eFrequency: %{y:0.0f}\",\"legendgroup\":\"Heart Disease\",\"marker\":{\"color\":\"#636efa\",\"pattern\":{\"shape\":\"\"}},\"name\":\"Heart Disease\",\"offsetgroup\":\"Heart Disease\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[17.0],\"textfont\":{\"color\":\"#fff\",\"family\":\"consolas\",\"size\":18},\"textposition\":\"auto\",\"x\":[\"Heart Disease\"],\"xaxis\":\"x\",\"y\":[17],\"yaxis\":\"y\",\"type\":\"bar\"},{\"alignmentgroup\":\"True\",\"hovertemplate\":\"%{x}\\u003cbr\\u003eFrequency: %{y:0.0f}\",\"legendgroup\":\"No Heart Disease\",\"marker\":{\"color\":\"#EF553B\",\"pattern\":{\"shape\":\"\"}},\"name\":\"No Heart Disease\",\"offsetgroup\":\"No Heart Disease\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[9.0],\"textfont\":{\"color\":\"#fff\",\"family\":\"consolas\",\"size\":18},\"textposition\":\"auto\",\"x\":[\"No Heart Disease\"],\"xaxis\":\"x\",\"y\":[9],\"yaxis\":\"y\",\"type\":\"bar\"}],                        {\"barmode\":\"relative\",\"hoverlabel\":{\"bgcolor\":\"#111\",\"font\":{\"family\":\"arial\",\"size\":16}},\"legend\":{\"title\":{\"text\":\"color\"},\"tracegroupgap\":0},\"showlegend\":true,\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"white\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"#C8D4E3\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"white\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"radialaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"yaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"zaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"caxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2}}},\"title\":{\"font\":{\"family\":\"tahoma\",\"size\":24},\"text\":\"Frequency of Heart Disease With Blood Pressure \\u003e 170\"},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Heart Disease\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Frequency in PCT(%)\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('c4350820-4570-42a8-841f-4e824bfd99e6');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "high_blood = df[df[\"RestingBP\"] > 170][\"HeartDisease\"].value_counts()\n",
    "\n",
    "\n",
    "\n",
    "fig = px.bar(\n",
    "    data_frame = high_blood,\n",
    "    x = [\"No Heart Disease\" if i == 0 else \"Heart Disease\" for i in high_blood.index],\n",
    "    y = high_blood,\n",
    "    color = [\"No Heart Disease\" if i == 0 else \"Heart Disease\" for i in high_blood.index],\n",
    "    title = \"Frequency of Heart Disease With Blood Pressure > 170\",\n",
    "    labels= {\"x\" :\"Heart Disease\", \"y\": \"Frequency in PCT(%)\"},\n",
    "    template=\"plotly_white\",\n",
    "    text = high_blood\n",
    ")\n",
    "\n",
    "custome_layout(title_size=24, showlegend=True) \n",
    "\n",
    "\n",
    "fig.update_traces(\n",
    "    textfont = {\n",
    "        \"size\" : 18,\n",
    "        \"family\" :\"consolas\",\n",
    "        \"color\": \"#fff\"\n",
    "    },\n",
    "    hovertemplate = \"%{x}<br>Frequency: %{y:0.0f}\",\n",
    ")\n",
    "\n",
    "iplot(fig)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "96057b36",
   "metadata": {
    "papermill": {
     "duration": 0.040582,
     "end_time": "2024-03-10T10:04:48.712392",
     "exception": false,
     "start_time": "2024-03-10T10:04:48.671810",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<h2 style=\"font: bold 26px tahoma\">\n",
    "    ♠ Cholesterol Column \n",
    "</h2>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "e55f8b87",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:48.796015Z",
     "iopub.status.busy": "2024-03-10T10:04:48.795567Z",
     "iopub.status.idle": "2024-03-10T10:04:48.894140Z",
     "shell.execute_reply": "2024-03-10T10:04:48.892837Z"
    },
    "papermill": {
     "duration": 0.14361,
     "end_time": "2024-03-10T10:04:48.896726",
     "exception": false,
     "start_time": "2024-03-10T10:04:48.753116",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>                            <div id=\"04ef39fd-b547-4bbc-8771-1cbf8de39139\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"04ef39fd-b547-4bbc-8771-1cbf8de39139\")) {                    Plotly.newPlot(                        \"04ef39fd-b547-4bbc-8771-1cbf8de39139\",                        [{\"alignmentgroup\":\"True\",\"hovertemplate\":\"Cholesterol=%{x}\\u003cextra\\u003e\\u003c\\u002fextra\\u003e\",\"legendgroup\":\"\",\"marker\":{\"color\":\"#636efa\"},\"name\":\"\",\"notched\":false,\"offsetgroup\":\"\",\"orientation\":\"h\",\"showlegend\":false,\"x\":[289,180,283,214,195,339,237,208,207,284,211,164,204,234,211,273,196,201,248,267,223,184,201,288,215,209,260,284,468,188,518,167,224,172,186,254,306,250,177,227,230,294,264,259,175,318,223,216,340,289,233,205,224,245,180,194,270,213,365,342,253,254,224,277,202,260,297,225,246,412,265,215,182,218,268,163,529,167,100,206,277,238,223,196,213,139,263,216,291,229,208,307,210,329,182,263,207,147,85,269,275,179,392,466,186,260,254,214,129,241,188,255,276,297,207,246,282,338,160,156,248,272,240,393,230,246,161,163,230,228,292,202,388,230,294,265,215,241,166,247,331,341,291,243,279,273,198,249,168,603,215,159,275,270,291,342,190,185,290,195,264,212,263,196,225,272,231,238,222,179,243,235,320,187,266,288,216,287,194,238,225,224,404,238,312,211,251,237,328,285,280,209,245,192,184,193,297,268,246,308,249,230,147,219,184,215,308,257,132,216,263,288,276,219,226,237,280,217,196,263,222,303,195,298,256,264,195,117,295,173,315,281,275,250,309,200,336,295,355,193,326,198,292,266,268,171,237,275,219,341,491,260,292,271,248,274,394,160,200,320,275,221,231,126,193,305,298,220,242,235,225,198,201,220,295,213,160,223,347,253,246,222,220,344,358,190,169,181,308,166,211,257,182,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,260,209,218,228,213,0,236,0,0,267,166,0,0,0,0,220,177,236,0,0,0,0,0,0,0,0,0,186,100,228,0,171,230,0,0,0,281,0,203,0,0,0,0,0,277,0,233,0,0,240,0,0,153,224,0,0,0,316,0,0,218,0,311,0,0,0,270,0,0,217,214,214,252,220,214,203,0,339,216,276,458,241,384,297,248,308,208,227,210,245,225,240,0,198,195,267,161,258,0,0,195,235,0,305,223,282,349,160,160,236,312,283,142,211,218,306,186,252,222,0,0,258,202,197,204,113,274,192,298,272,220,200,261,181,260,220,221,216,175,219,310,208,232,273,203,182,274,204,270,292,171,221,289,217,223,110,193,123,210,282,170,369,173,289,152,208,216,271,244,285,243,240,219,237,165,213,287,258,256,186,264,185,226,203,207,284,337,310,254,258,254,300,170,310,333,139,223,385,254,322,564,261,263,269,177,256,239,293,407,234,226,235,234,303,149,311,203,211,199,229,245,303,204,288,275,243,295,230,265,229,228,215,326,200,256,207,273,180,222,223,209,233,197,218,211,149,197,246,225,315,205,417,195,234,198,166,178,249,281,126,305,226,240,233,276,261,319,242,243,260,354,245,197,223,309,208,199,209,236,218,198,270,214,201,244,208,270,306,243,221,330,266,206,212,275,302,234,313,244,141,237,269,289,254,274,222,258,177,160,327,235,305,304,295,271,249,288,226,283,188,286,274,360,273,201,267,196,201,230,269,212,226,246,232,177,277,249,210,207,212,271,233,213,283,282,230,167,224,268,250,219,267,303,256,204,217,308,193,228,231,244,262,259,211,325,254,197,236,282,234,254,299,211,182,294,298,231,254,196,240,409,172,265,246,315,184,233,394,269,239,174,309,282,255,250,248,214,239,304,277,300,258,299,289,298,318,240,309,250,288,245,213,216,204,204,252,227,258,220,239,254,168,330,183,203,263,341,283,186,307,219,260,255,231,164,234,177,257,325,274,321,264,268,308,253,248,269,185,282,188,219,290,175,212,302,243,353,335,247,340,206,284,266,229,199,263,294,192,286,216,223,247,204,204,227,278,220,232,197,335,253,205,192,203,318,225,220,221,240,212,342,169,187,197,157,176,241,264,193,131,236,175],\"x0\":\" \",\"xaxis\":\"x\",\"y0\":\" \",\"yaxis\":\"y\",\"type\":\"box\"}],                        {\"boxmode\":\"group\",\"hoverlabel\":{\"bgcolor\":\"#111\",\"font\":{\"family\":\"arial\",\"size\":16}},\"legend\":{\"tracegroupgap\":0},\"showlegend\":false,\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"white\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"#C8D4E3\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"white\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"radialaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"yaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"zaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"caxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2}}},\"title\":{\"font\":{\"family\":\"tahoma\",\"size\":28},\"text\":\"Cholesterol Distribution\"},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Cholesterol\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0]}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('04ef39fd-b547-4bbc-8771-1cbf8de39139');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = px.box(\n",
    "    x = df[\"Cholesterol\"], \n",
    "    title= \"Cholesterol Distribution\",\n",
    "    template=\"plotly_white\",\n",
    "    labels={\"x\": \"Cholesterol\"}\n",
    ")\n",
    "\n",
    "custome_layout()\n",
    "\n",
    "\n",
    "iplot(fig)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "89c990f9",
   "metadata": {
    "papermill": {
     "duration": 0.040438,
     "end_time": "2024-03-10T10:04:48.977522",
     "exception": false,
     "start_time": "2024-03-10T10:04:48.937084",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<p style = \"color: #F9F07A;\n",
    "            font: bold 17px arial;\n",
    "            background-color: #111;\n",
    "            padding: 20px;\n",
    "            border: 3px solid #FFF67E;\n",
    "            border-radius: 8px\"> \n",
    "    • After a little search, i found that, High levels of cholesterol in the blood can increase the risk of heart disease.\n",
    "    <br>\n",
    "    <br>\n",
    "    ♠ So, let's check check the heart disease for those who have high cholesterol in the blood greater than 407 mg/dl (upper)\n",
    "</p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "86087817",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:49.061646Z",
     "iopub.status.busy": "2024-03-10T10:04:49.061249Z",
     "iopub.status.idle": "2024-03-10T10:04:49.071262Z",
     "shell.execute_reply": "2024-03-10T10:04:49.069610Z"
    },
    "papermill": {
     "duration": 0.055217,
     "end_time": "2024-03-10T10:04:49.074038",
     "exception": false,
     "start_time": "2024-03-10T10:04:49.018821",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "HeartDisease\n",
       "1    6\n",
       "0    5\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "chol = df[df[\"Cholesterol\"] > 407][\"HeartDisease\"].value_counts()\n",
    "chol"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0809c670",
   "metadata": {
    "papermill": {
     "duration": 0.040249,
     "end_time": "2024-03-10T10:04:49.155573",
     "exception": false,
     "start_time": "2024-03-10T10:04:49.115324",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<h2 style=\"font: bold 26px tahoma\">\n",
    "    ♠ Fasting Blood Sugar Column 💉\n",
    "</h2>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "aef3c879",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:49.240861Z",
     "iopub.status.busy": "2024-03-10T10:04:49.240388Z",
     "iopub.status.idle": "2024-03-10T10:04:49.253068Z",
     "shell.execute_reply": "2024-03-10T10:04:49.251726Z"
    },
    "papermill": {
     "duration": 0.059484,
     "end_time": "2024-03-10T10:04:49.255969",
     "exception": false,
     "start_time": "2024-03-10T10:04:49.196485",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "FastingBS\n",
       "0    76.69%\n",
       "1    23.31%\n",
       "Name: proportion, dtype: object"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# ===== To Know What the Fasting Blood Sugar  is, Go To The Columns Details 💊 Section =====\n",
    "blood_sugar = df[\"FastingBS\"].value_counts(normalize=1) * 100\n",
    "blood_sugar.apply(lambda x: f\"{x:0.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "94a9eac1",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:49.342467Z",
     "iopub.status.busy": "2024-03-10T10:04:49.340553Z",
     "iopub.status.idle": "2024-03-10T10:04:49.466968Z",
     "shell.execute_reply": "2024-03-10T10:04:49.465682Z"
    },
    "papermill": {
     "duration": 0.172155,
     "end_time": "2024-03-10T10:04:49.469730",
     "exception": false,
     "start_time": "2024-03-10T10:04:49.297575",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>                            <div id=\"453e2f93-86be-42af-9f0f-2b5650c0a52c\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"453e2f93-86be-42af-9f0f-2b5650c0a52c\")) {                    Plotly.newPlot(                        \"453e2f93-86be-42af-9f0f-2b5650c0a52c\",                        [{\"alignmentgroup\":\"True\",\"hovertemplate\":\"Fasting Blood Sugar: %{x}\\u003cbr\\u003ePercentage: %{y:0.2f}%\",\"legendgroup\":\"0\",\"marker\":{\"color\":\"#636efa\",\"pattern\":{\"shape\":\"\"}},\"name\":\"0\",\"offsetgroup\":\"0\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"77%\"],\"textfont\":{\"color\":\"#fff\",\"family\":\"consolas\",\"size\":18},\"textposition\":\"auto\",\"x\":[\"Less Than 120\"],\"xaxis\":\"x\",\"y\":[76.68845315904139],\"yaxis\":\"y\",\"type\":\"bar\"},{\"alignmentgroup\":\"True\",\"hovertemplate\":\"Fasting Blood Sugar: %{x}\\u003cbr\\u003ePercentage: %{y:0.2f}%\",\"legendgroup\":\"1\",\"marker\":{\"color\":\"#EF553B\",\"pattern\":{\"shape\":\"\"}},\"name\":\"1\",\"offsetgroup\":\"1\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"23%\"],\"textfont\":{\"color\":\"#fff\",\"family\":\"consolas\",\"size\":18},\"textposition\":\"auto\",\"x\":[\"Greate than 120\"],\"xaxis\":\"x\",\"y\":[23.311546840958606],\"yaxis\":\"y\",\"type\":\"bar\"}],                        {\"barmode\":\"relative\",\"hoverlabel\":{\"bgcolor\":\"#111\",\"font\":{\"family\":\"arial\",\"size\":16}},\"legend\":{\"title\":{\"text\":\"color\"},\"tracegroupgap\":0},\"showlegend\":false,\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"white\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"#C8D4E3\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"white\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"radialaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"yaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"zaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"caxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2}}},\"title\":{\"font\":{\"family\":\"tahoma\",\"size\":28},\"text\":\"Fasting Blood Sugar Frequency 🩸\"},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Fasting Blood Sugar\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Frequency in PCT(%)\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('453e2f93-86be-42af-9f0f-2b5650c0a52c');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = count_bar_plot(\n",
    "    data_frame=df, column_name=\"FastingBS\", x_title= \"Fasting Blood Sugar\", y_title=\"Frequency in PCT(%)\", \n",
    "    title=\"Fasting Blood Sugar Frequency 🩸\",\n",
    "    hover_template=\"Fasting Blood Sugar: %{x}<br>Percentage: %{y:0.2f}%\",\n",
    "    bars_names=[\"Greate than 120\" if i == 1 else \"Less Than 120\" for i in blood_sugar.index]\n",
    ")\n",
    "iplot(fig)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f4ed0cc7",
   "metadata": {
    "papermill": {
     "duration": 0.042337,
     "end_time": "2024-03-10T10:04:49.556001",
     "exception": false,
     "start_time": "2024-03-10T10:04:49.513664",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<h2 style=\"font: bold 26px tahoma\">\n",
    "    ♠ ECG Results Column 📈\n",
    "</h2>"
   ]
  },
  {
   "attachments": {
    "image.png": {
     "image/png": "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"
    }
   },
   "cell_type": "markdown",
   "id": "d73ffca5",
   "metadata": {
    "papermill": {
     "duration": 0.041643,
     "end_time": "2024-03-10T10:04:49.639055",
     "exception": false,
     "start_time": "2024-03-10T10:04:49.597412",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "![image.png](attachment:image.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "0b4c2b67",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:49.724283Z",
     "iopub.status.busy": "2024-03-10T10:04:49.723520Z",
     "iopub.status.idle": "2024-03-10T10:04:49.734753Z",
     "shell.execute_reply": "2024-03-10T10:04:49.733637Z"
    },
    "papermill": {
     "duration": 0.057056,
     "end_time": "2024-03-10T10:04:49.737125",
     "exception": false,
     "start_time": "2024-03-10T10:04:49.680069",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RestingECG\n",
       "Normal    60.13%\n",
       "LVH       20.48%\n",
       "ST        19.39%\n",
       "Name: proportion, dtype: object"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# ===== To Know What the RestingECG is, Go To The Columns Details 💊 Section =====\n",
    "ecg = df[\"RestingECG\"].value_counts(normalize=1) * 100\n",
    "ecg.apply(lambda x: f\"{x:0.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "3a58f1cd",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:49.820348Z",
     "iopub.status.busy": "2024-03-10T10:04:49.818900Z",
     "iopub.status.idle": "2024-03-10T10:04:49.957010Z",
     "shell.execute_reply": "2024-03-10T10:04:49.955758Z"
    },
    "papermill": {
     "duration": 0.183151,
     "end_time": "2024-03-10T10:04:49.960113",
     "exception": false,
     "start_time": "2024-03-10T10:04:49.776962",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>                            <div id=\"cfb9701a-71b3-4353-88cf-aa67f78013cf\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"cfb9701a-71b3-4353-88cf-aa67f78013cf\")) {                    Plotly.newPlot(                        \"cfb9701a-71b3-4353-88cf-aa67f78013cf\",                        [{\"alignmentgroup\":\"True\",\"hovertemplate\":\"ECG Result: %{x}\\u003cbr\\u003eFrequency Percentaeg: %{y:0.2f}%\",\"legendgroup\":\"Normal\",\"marker\":{\"color\":\"#636efa\",\"pattern\":{\"shape\":\"\"}},\"name\":\"Normal\",\"offsetgroup\":\"Normal\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"60%\"],\"textfont\":{\"color\":\"#fff\",\"family\":\"consolas\",\"size\":18},\"textposition\":\"auto\",\"x\":[\"Normal\"],\"xaxis\":\"x\",\"y\":[60.130718954248366],\"yaxis\":\"y\",\"type\":\"bar\"},{\"alignmentgroup\":\"True\",\"hovertemplate\":\"ECG Result: %{x}\\u003cbr\\u003eFrequency Percentaeg: %{y:0.2f}%\",\"legendgroup\":\"LVH\",\"marker\":{\"color\":\"#EF553B\",\"pattern\":{\"shape\":\"\"}},\"name\":\"LVH\",\"offsetgroup\":\"LVH\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"20%\"],\"textfont\":{\"color\":\"#fff\",\"family\":\"consolas\",\"size\":18},\"textposition\":\"auto\",\"x\":[\"LVH\"],\"xaxis\":\"x\",\"y\":[20.47930283224401],\"yaxis\":\"y\",\"type\":\"bar\"},{\"alignmentgroup\":\"True\",\"hovertemplate\":\"ECG Result: %{x}\\u003cbr\\u003eFrequency Percentaeg: %{y:0.2f}%\",\"legendgroup\":\"ST\",\"marker\":{\"color\":\"#00cc96\",\"pattern\":{\"shape\":\"\"}},\"name\":\"ST\",\"offsetgroup\":\"ST\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"19%\"],\"textfont\":{\"color\":\"#fff\",\"family\":\"consolas\",\"size\":18},\"textposition\":\"auto\",\"x\":[\"ST\"],\"xaxis\":\"x\",\"y\":[19.389978213507625],\"yaxis\":\"y\",\"type\":\"bar\"}],                        {\"barmode\":\"relative\",\"hoverlabel\":{\"bgcolor\":\"#111\",\"font\":{\"family\":\"arial\",\"size\":16}},\"legend\":{\"title\":{\"text\":\"color\"},\"tracegroupgap\":0},\"showlegend\":false,\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"white\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"#C8D4E3\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"white\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"radialaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"yaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"zaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"caxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2}}},\"title\":{\"font\":{\"family\":\"tahoma\",\"size\":28},\"text\":\"ECG Results Frequency 📈\"},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"RestingECG\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Frequency in PCT(%)\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('cfb9701a-71b3-4353-88cf-aa67f78013cf');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = count_bar_plot(\n",
    "    data_frame=df, column_name=\"RestingECG\", x_title= \"ECG Results\", y_title=\"Frequency in PCT(%)\", \n",
    "    title=\"ECG Results Frequency 📈\",\n",
    "    hover_template=\"ECG Result: %{x}<br>Frequency Percentaeg: %{y:0.2f}%\",\n",
    ")\n",
    "iplot(fig)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2097eca9",
   "metadata": {
    "papermill": {
     "duration": 0.042113,
     "end_time": "2024-03-10T10:04:50.043046",
     "exception": false,
     "start_time": "2024-03-10T10:04:50.000933",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<h2 style=\"font: bold 26px tahoma\">\n",
    "    ♠ Max Heart Rate Column ❤️🩺\n",
    "</h2>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "2106f739",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:50.128877Z",
     "iopub.status.busy": "2024-03-10T10:04:50.128403Z",
     "iopub.status.idle": "2024-03-10T10:04:50.232886Z",
     "shell.execute_reply": "2024-03-10T10:04:50.231580Z"
    },
    "papermill": {
     "duration": 0.150686,
     "end_time": "2024-03-10T10:04:50.235534",
     "exception": false,
     "start_time": "2024-03-10T10:04:50.084848",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>                            <div id=\"97957c56-8ff5-4978-ac23-3f8ad9524809\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"97957c56-8ff5-4978-ac23-3f8ad9524809\")) {                    Plotly.newPlot(                        \"97957c56-8ff5-4978-ac23-3f8ad9524809\",                        [{\"alignmentgroup\":\"True\",\"hovertemplate\":\"Max Heart Rate=%{x}\\u003cextra\\u003e\\u003c\\u002fextra\\u003e\",\"legendgroup\":\"\",\"marker\":{\"color\":\"#636efa\"},\"name\":\"\",\"notched\":false,\"offsetgroup\":\"\",\"orientation\":\"h\",\"showlegend\":false,\"x\":[172,156,98,108,122,170,170,142,130,120,142,99,145,140,137,150,166,165,125,160,142,142,164,150,138,178,112,118,127,145,130,114,122,130,154,155,87,142,148,130,130,100,168,170,120,120,168,170,184,170,121,98,122,150,140,170,153,140,134,96,174,175,144,125,145,130,144,184,82,170,145,135,150,115,128,116,130,150,138,170,160,154,115,165,125,94,112,142,155,110,160,140,148,92,180,140,138,160,140,144,115,100,130,152,124,140,110,168,135,106,124,92,125,150,135,150,170,130,185,180,170,139,140,110,150,110,190,175,140,152,130,150,122,124,120,175,175,146,118,130,94,125,158,155,150,132,155,176,160,125,120,100,150,140,160,150,150,130,100,130,119,96,174,150,140,175,140,118,100,160,160,188,162,172,134,135,105,150,150,90,120,150,124,140,130,92,110,138,110,120,120,116,160,110,180,116,132,136,116,98,150,150,146,150,100,140,180,140,185,140,110,140,128,164,98,170,150,137,150,170,112,150,125,185,137,150,140,134,170,184,158,167,129,142,140,160,118,136,99,102,155,142,143,118,103,137,150,150,130,120,135,115,115,152,96,130,150,172,120,155,165,138,115,125,145,175,110,150,91,145,140,165,130,134,180,100,150,126,126,155,135,122,160,160,170,120,140,132,156,180,138,135,148,93,127,110,139,131,92,149,149,150,120,123,126,127,155,120,138,182,154,110,176,154,141,123,148,121,77,136,175,109,166,128,133,128,138,119,82,130,143,82,179,144,170,134,114,154,149,145,122,114,113,120,104,130,115,128,104,125,120,140,100,100,92,125,113,95,128,115,72,124,99,148,97,140,117,120,120,86,63,108,98,115,105,121,118,122,157,156,99,120,145,156,155,105,99,135,83,145,60,92,115,120,98,150,143,105,122,70,110,163,67,128,120,130,100,72,94,122,78,150,103,98,110,90,112,127,140,149,99,120,105,140,141,157,140,117,120,120,148,86,84,125,120,118,124,106,111,116,180,129,125,140,120,124,117,110,105,155,110,122,118,133,123,131,80,165,86,111,118,84,117,107,128,160,125,130,97,161,106,130,140,122,130,120,139,108,148,123,110,118,125,106,112,128,180,144,135,140,102,108,145,127,110,140,69,148,130,130,140,138,140,138,112,131,112,80,150,110,126,88,153,150,120,160,132,120,110,121,128,135,120,117,150,144,113,135,127,109,128,115,102,140,135,122,119,130,112,100,122,120,105,129,120,139,162,100,140,135,73,86,108,116,160,118,112,122,124,102,137,141,154,126,160,115,128,115,105,110,119,109,135,130,112,126,120,110,119,110,130,159,84,126,116,120,122,165,122,94,133,110,150,130,113,140,100,136,127,98,96,123,98,112,151,96,108,128,138,126,154,137,100,135,93,109,160,141,105,121,140,142,142,170,154,161,111,180,145,159,125,120,155,144,178,129,180,181,143,159,139,152,157,165,130,150,138,170,140,126,150,138,125,150,186,181,163,179,156,134,165,126,177,120,114,125,184,157,179,175,168,125,96,143,103,173,142,169,171,150,112,186,152,149,152,140,163,143,116,142,147,148,179,173,178,105,130,111,168,126,178,140,145,163,128,164,169,109,108,168,118,151,156,133,162,175,71,163,124,147,166,143,157,162,138,117,153,161,170,162,162,144,133,114,103,139,116,88,151,152,163,99,169,158,160,169,132,178,96,165,160,172,144,192,168,132,182,163,125,195,95,160,114,173,172,179,158,167,122,149,172,111,170,162,165,182,154,155,130,161,154,159,152,152,174,131,146,125,115,174,106,122,147,163,163,194,150,158,122,173,162,105,147,157,112,160,125,156,156,175,161,122,158,151,162,151,171,141,173,145,178,160,154,131,187,159,166,165,131,202,172,172,154,147,170,126,127,174,132,182,132,97,136,162,190,146,140,185,161,146,145,160,120,156,172,150,182,143,160,142,144,158,148,155,142,113,188,153,123,157,162,137,132,158,171,172,132,160,171,168,162,173,153,148,108,115,169,143,156,162,155,152,152,164,131,143,179,130,174,161,140,146,144,163,169,150,166,144,144,136,182,90,123,132,141,115,174,173],\"x0\":\" \",\"xaxis\":\"x\",\"y0\":\" \",\"yaxis\":\"y\",\"type\":\"box\"}],                        {\"boxmode\":\"group\",\"hoverlabel\":{\"bgcolor\":\"#111\",\"font\":{\"family\":\"arial\",\"size\":16}},\"legend\":{\"tracegroupgap\":0},\"showlegend\":false,\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"white\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"#C8D4E3\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"white\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"radialaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"yaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"zaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"caxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2}}},\"title\":{\"font\":{\"family\":\"tahoma\",\"size\":28},\"text\":\"Max Heart Rate Distribution\"},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Max Heart Rate\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0]}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('97957c56-8ff5-4978-ac23-3f8ad9524809');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = px.box(\n",
    "    x = df[\"MaxHR\"], \n",
    "    title= \"Max Heart Rate Distribution\",\n",
    "    template=\"plotly_white\",\n",
    "    labels={\"x\": \"Max Heart Rate\"}\n",
    ")\n",
    "\n",
    "custome_layout()\n",
    "\n",
    "\n",
    "iplot(fig)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ade00546",
   "metadata": {
    "papermill": {
     "duration": 0.043186,
     "end_time": "2024-03-10T10:04:50.321023",
     "exception": false,
     "start_time": "2024-03-10T10:04:50.277837",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<p style = \"color: #F9F07A;\n",
    "            font: bold 17px arial;\n",
    "            background-color: #111;\n",
    "            padding: 20px;\n",
    "            border: 3px solid #FFF67E;\n",
    "            border-radius: 8px\"> \n",
    "    • After Googling, i found that, Lower values might indicate heart disease or poor heart health...\n",
    "    <br>\n",
    "    <br>\n",
    "    ♠ So, let's check check the heart disease for those who have low Max Heart Rate less than 67 (lower)\n",
    "</p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "06303776",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:50.406322Z",
     "iopub.status.busy": "2024-03-10T10:04:50.405560Z",
     "iopub.status.idle": "2024-03-10T10:04:50.421917Z",
     "shell.execute_reply": "2024-03-10T10:04:50.421038Z"
    },
    "papermill": {
     "duration": 0.061433,
     "end_time": "2024-03-10T10:04:50.424225",
     "exception": false,
     "start_time": "2024-03-10T10:04:50.362792",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Sex</th>\n",
       "      <th>ChestPainType</th>\n",
       "      <th>RestingBP</th>\n",
       "      <th>Cholesterol</th>\n",
       "      <th>FastingBS</th>\n",
       "      <th>RestingECG</th>\n",
       "      <th>MaxHR</th>\n",
       "      <th>ExerciseAngina</th>\n",
       "      <th>Oldpeak</th>\n",
       "      <th>ST_Slope</th>\n",
       "      <th>HeartDisease</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>370</th>\n",
       "      <td>60</td>\n",
       "      <td>M</td>\n",
       "      <td>ASY</td>\n",
       "      <td>135</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Normal</td>\n",
       "      <td>63</td>\n",
       "      <td>Y</td>\n",
       "      <td>0.5</td>\n",
       "      <td>Up</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>390</th>\n",
       "      <td>51</td>\n",
       "      <td>M</td>\n",
       "      <td>ASY</td>\n",
       "      <td>140</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Normal</td>\n",
       "      <td>60</td>\n",
       "      <td>N</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Flat</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Age Sex ChestPainType  RestingBP  Cholesterol  FastingBS RestingECG  \\\n",
       "370   60   M           ASY        135            0          0     Normal   \n",
       "390   51   M           ASY        140            0          0     Normal   \n",
       "\n",
       "     MaxHR ExerciseAngina  Oldpeak ST_Slope  HeartDisease  \n",
       "370     63              Y      0.5       Up             1  \n",
       "390     60              N      0.0     Flat             1  "
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df[\"MaxHR\"] < 67]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a1697126",
   "metadata": {
    "papermill": {
     "duration": 0.042383,
     "end_time": "2024-03-10T10:04:50.508178",
     "exception": false,
     "start_time": "2024-03-10T10:04:50.465795",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<h2 style=\"font: bold 26px tahoma\">\n",
    "    ♠ Exercise Angina Column \n",
    "</h2>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "019e9c9d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:50.595360Z",
     "iopub.status.busy": "2024-03-10T10:04:50.594701Z",
     "iopub.status.idle": "2024-03-10T10:04:50.606500Z",
     "shell.execute_reply": "2024-03-10T10:04:50.605309Z"
    },
    "papermill": {
     "duration": 0.058168,
     "end_time": "2024-03-10T10:04:50.609021",
     "exception": false,
     "start_time": "2024-03-10T10:04:50.550853",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ExerciseAngina\n",
       "N    59.59%\n",
       "Y    40.41%\n",
       "Name: proportion, dtype: object"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# ===== To Know What the RestingECG is, Go To The Columns Details 💊 Section =====\n",
    "angina = df[\"ExerciseAngina\"].value_counts(normalize=1) * 100\n",
    "angina.apply(lambda x: f\"{x:0.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "22294337",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:50.697147Z",
     "iopub.status.busy": "2024-03-10T10:04:50.695544Z",
     "iopub.status.idle": "2024-03-10T10:04:50.823091Z",
     "shell.execute_reply": "2024-03-10T10:04:50.821875Z"
    },
    "papermill": {
     "duration": 0.175527,
     "end_time": "2024-03-10T10:04:50.825763",
     "exception": false,
     "start_time": "2024-03-10T10:04:50.650236",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>                            <div id=\"e9f61880-3764-45be-919a-a2a659af9d2f\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"e9f61880-3764-45be-919a-a2a659af9d2f\")) {                    Plotly.newPlot(                        \"e9f61880-3764-45be-919a-a2a659af9d2f\",                        [{\"alignmentgroup\":\"True\",\"hovertemplate\":\"Get Exercise Angina: %{x}\\u003cbr\\u003eFrequency Percentaeg: %{y:0.2f}%\",\"legendgroup\":\"N\",\"marker\":{\"color\":\"#636efa\",\"pattern\":{\"shape\":\"\"}},\"name\":\"N\",\"offsetgroup\":\"N\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"60%\"],\"textfont\":{\"color\":\"#fff\",\"family\":\"consolas\",\"size\":18},\"textposition\":\"auto\",\"x\":[\"No\"],\"xaxis\":\"x\",\"y\":[59.58605664488017],\"yaxis\":\"y\",\"type\":\"bar\"},{\"alignmentgroup\":\"True\",\"hovertemplate\":\"Get Exercise Angina: %{x}\\u003cbr\\u003eFrequency Percentaeg: %{y:0.2f}%\",\"legendgroup\":\"Y\",\"marker\":{\"color\":\"#EF553B\",\"pattern\":{\"shape\":\"\"}},\"name\":\"Y\",\"offsetgroup\":\"Y\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"40%\"],\"textfont\":{\"color\":\"#fff\",\"family\":\"consolas\",\"size\":18},\"textposition\":\"auto\",\"x\":[\"Yes\"],\"xaxis\":\"x\",\"y\":[40.41394335511983],\"yaxis\":\"y\",\"type\":\"bar\"}],                        {\"barmode\":\"relative\",\"hoverlabel\":{\"bgcolor\":\"#111\",\"font\":{\"family\":\"arial\",\"size\":16}},\"legend\":{\"title\":{\"text\":\"color\"},\"tracegroupgap\":0},\"showlegend\":false,\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"white\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"#C8D4E3\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"white\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"radialaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"yaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"zaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"caxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2}}},\"title\":{\"font\":{\"family\":\"tahoma\",\"size\":28},\"text\":\"Exercise Angina Frequency 📈\"},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Exercise Angina\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Frequency in PCT(%)\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('e9f61880-3764-45be-919a-a2a659af9d2f');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = count_bar_plot(\n",
    "    data_frame=df, column_name=\"ExerciseAngina\", x_title= \"Exercise Angina\", y_title=\"Frequency in PCT(%)\", \n",
    "    title=\"Exercise Angina Frequency 📈\",\n",
    "    hover_template=\"Get Exercise Angina: %{x}<br>Frequency Percentaeg: %{y:0.2f}%\",\n",
    "    bars_names=[\"Yes\" if i == 'Y' else \"No\" for i in angina.index]\n",
    ")\n",
    "iplot(fig)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b07b951b",
   "metadata": {
    "papermill": {
     "duration": 0.043709,
     "end_time": "2024-03-10T10:04:50.912450",
     "exception": false,
     "start_time": "2024-03-10T10:04:50.868741",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<h2 style=\"font: bold 26px tahoma\">\n",
    "    ♠ Oldpeak Column \n",
    "</h2>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "c4c8af5f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:51.001971Z",
     "iopub.status.busy": "2024-03-10T10:04:51.001207Z",
     "iopub.status.idle": "2024-03-10T10:04:51.102812Z",
     "shell.execute_reply": "2024-03-10T10:04:51.101536Z"
    },
    "papermill": {
     "duration": 0.149473,
     "end_time": "2024-03-10T10:04:51.105586",
     "exception": false,
     "start_time": "2024-03-10T10:04:50.956113",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>                            <div id=\"d6ba2bf0-7e5f-42cc-8993-5bb616560e1b\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"d6ba2bf0-7e5f-42cc-8993-5bb616560e1b\")) {                    Plotly.newPlot(                        \"d6ba2bf0-7e5f-42cc-8993-5bb616560e1b\",                        [{\"alignmentgroup\":\"True\",\"hovertemplate\":\"Oldpeak=%{x}\\u003cextra\\u003e\\u003c\\u002fextra\\u003e\",\"legendgroup\":\"\",\"marker\":{\"color\":\"#636efa\"},\"name\":\"\",\"notched\":false,\"offsetgroup\":\"\",\"orientation\":\"h\",\"showlegend\":false,\"x\":[0.0,1.0,0.0,1.5,0.0,0.0,0.0,0.0,1.5,0.0,0.0,2.0,0.0,1.0,0.0,1.5,0.0,0.0,1.0,3.0,0.0,1.0,0.0,3.0,0.0,0.0,3.0,0.0,0.0,0.0,0.0,0.0,2.0,2.0,0.0,0.0,1.5,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,2.0,2.0,0.0,0.0,1.5,0.0,1.5,0.0,1.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,4.0,0.0,1.0,0.0,0.0,0.0,1.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,2.0,2.0,0.0,0.5,0.0,0.0,0.0,1.5,0.0,2.0,0.0,0.0,0.0,0.0,1.0,0.0,2.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,3.0,0.0,0.0,0.0,1.0,0.0,1.5,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,2.0,0.0,1.5,0.0,0.0,2.0,1.5,1.0,0.0,0.0,2.0,0.0,2.0,2.5,2.5,3.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,3.0,1.0,0.0,2.0,1.0,0.0,0.0,0.0,0.0,0.0,2.0,5.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,2.0,1.5,0.0,0.0,0.0,2.0,0.0,2.0,1.0,0.0,0.0,0.0,1.0,1.0,1.5,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.5,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,2.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,3.0,0.0,2.0,3.0,0.0,2.0,2.0,0.0,1.0,2.0,1.5,2.0,1.0,1.0,0.0,2.0,0.0,1.0,2.0,0.0,0.0,0.0,0.5,0.0,0.0,1.0,0.0,0.0,1.0,0.0,1.0,0.0,1.0,2.0,0.0,0.0,3.0,0.0,0.0,0.0,2.0,1.5,0.8,0.0,0.0,2.0,2.0,0.0,0.0,0.0,0.0,0.0,2.0,0.0,0.0,1.0,0.0,0.0,0.7,1.5,0.7,1.4,0.0,2.1,0.4,0.2,1.5,1.7,2.2,1.5,0.1,0.7,0.5,0.7,1.0,0.1,1.6,0.2,2.0,1.3,0.3,1.8,2.5,1.8,2.6,-0.9,2.8,2.5,-2.6,-1.5,-0.1,0.9,0.8,1.1,2.4,-1.0,-1.1,0.0,-0.7,-0.8,1.6,3.7,2.0,1.1,1.5,1.3,1.4,0.0,0.0,0.0,0.0,0.0,1.6,1.0,0.0,0.5,-1.0,1.0,0.3,0.0,1.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,2.0,0.0,2.0,2.0,0.5,2.0,0.0,1.0,0.0,0.0,1.0,1.2,2.0,0.0,0.5,0.5,2.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.7,2.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.7,2.0,0.0,1.2,0.0,-0.5,0.0,0.0,2.0,1.5,1.0,-2.0,3.0,0.0,3.0,0.0,1.5,2.5,1.3,-0.5,0.0,1.5,2.0,0.5,0.0,1.0,0.5,1.0,1.0,0.0,2.5,2.0,1.5,0.0,1.0,2.0,0.0,0.2,3.0,1.0,1.2,0.5,1.5,1.6,1.4,2.0,1.0,1.5,2.0,1.0,1.5,2.0,1.2,1.5,0.0,0.0,1.5,0.0,1.9,0.0,1.3,0.0,2.0,0.0,2.5,0.1,1.6,2.0,0.0,3.0,1.5,1.7,0.1,0.0,0.1,2.0,2.0,2.5,2.0,2.5,2.5,1.5,1.1,1.2,0.4,2.0,0.3,3.0,1.0,0.0,3.0,1.7,2.5,1.0,1.0,3.0,0.0,1.0,4.0,2.0,2.0,0.2,3.0,1.2,3.0,0.0,1.5,0.0,0.3,2.0,-0.1,1.3,0.5,3.0,0.0,1.5,1.0,1.0,0.5,4.0,1.0,1.0,0.0,0.1,1.7,0.3,1.5,1.4,1.1,1.8,0.0,2.0,2.5,1.0,1.2,4.0,2.0,0.0,1.2,3.5,1.5,3.0,0.0,0.2,0.0,1.5,1.5,0.2,2.0,0.0,1.8,1.8,0.3,0.0,2.0,1.8,1.4,4.0,0.2,0.1,2.0,1.1,2.0,1.7,1.5,0.0,1.5,2.5,2.0,1.5,0.5,1.5,1.5,1.2,3.0,1.9,3.0,1.8,1.0,1.5,0.0,0.3,1.5,0.8,2.0,1.0,2.0,0.0,0.2,0.0,2.0,0.0,1.0,0.5,0.0,0.2,1.7,1.5,1.0,1.3,0.0,1.5,0.0,1.0,3.0,1.5,0.0,0.0,0.0,0.2,0.0,0.3,0.0,2.4,1.6,0.3,0.2,0.2,0.4,0.6,1.2,1.2,4.0,0.5,0.0,0.0,2.6,0.0,1.6,1.8,3.1,1.8,1.4,2.6,0.2,1.2,0.1,0.0,0.2,0.0,0.6,2.5,0.0,0.4,2.3,0.0,3.4,0.9,0.0,1.9,0.0,0.0,0.0,0.0,0.0,0.4,0.0,2.2,0.0,0.8,0.0,0.0,1.0,1.8,0.0,0.8,0.0,0.6,0.0,3.6,0.0,0.0,1.4,0.2,1.2,0.0,0.9,2.3,0.6,0.0,0.0,0.3,0.0,3.6,0.6,0.0,1.1,0.3,0.0,3.0,0.0,0.0,0.8,2.0,1.6,0.8,2.0,1.5,0.8,0.0,4.2,0.0,2.6,0.0,0.0,2.2,0.0,1.0,1.0,0.4,0.1,0.2,1.1,0.6,1.0,0.0,1.0,1.4,0.5,1.2,2.6,0.0,0.0,3.4,0.0,0.0,0.0,0.0,0.0,0.8,4.0,2.6,1.6,2.0,3.2,1.2,0.8,0.5,0.0,1.8,0.1,0.8,1.4,1.8,0.1,0.0,2.2,1.6,1.4,0.0,1.2,0.7,0.0,2.0,0.0,0.6,1.4,0.0,2.0,0.0,2.0,3.2,0.0,0.0,1.6,0.0,2.0,0.5,0.0,5.6,0.0,1.9,1.0,3.8,1.4,0.0,3.0,0.0,0.0,0.0,1.2,0.2,1.4,0.1,2.0,0.9,1.5,0.0,1.9,4.2,3.6,0.2,0.0,0.8,1.9,0.0,0.6,0.0,1.9,2.1,0.1,1.2,2.9,1.2,2.6,0.0,0.0,0.0,1.4,1.0,1.6,1.8,0.0,1.0,0.0,2.8,1.6,0.8,1.2,0.0,0.6,1.8,3.5,0.2,2.4,0.2,2.2,0.0,1.4,0.0,0.0,0.4,0.0,2.8,2.8,1.6,1.8,1.4,0.0,1.2,3.0,1.0,0.0,1.0,1.2,0.0,0.0,1.8,6.2,0.0,2.5,0.0,0.2,1.6,0.0,0.4,3.6,1.5,1.4,0.6,0.8,3.0,2.8,1.4,0.0,0.0,0.6,1.6,0.4,1.0,1.2,0.0,1.5,0.0,2.4,1.8,0.6,1.0,0.5,0.0,1.3,0.4,1.5,0.0,0.0,0.1,1.0,0.8,0.6,0.0,0.0,0.0,0.6,3.0,0.0,2.0,0.0,0.0,4.4,2.8,0.4,0.0,0.0,0.8,1.2,2.8,4.0,0.0,0.0,1.0,0.2,1.2,3.4,1.2,0.0,0.0],\"x0\":\" \",\"xaxis\":\"x\",\"y0\":\" \",\"yaxis\":\"y\",\"type\":\"box\"}],                        {\"boxmode\":\"group\",\"hoverlabel\":{\"bgcolor\":\"#111\",\"font\":{\"family\":\"arial\",\"size\":16}},\"legend\":{\"tracegroupgap\":0},\"showlegend\":false,\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"white\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"#C8D4E3\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"white\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"radialaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"yaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"zaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"caxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2}}},\"title\":{\"font\":{\"family\":\"tahoma\",\"size\":28},\"text\":\"Oldpeak Distribution\"},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Oldpeak\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0]}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('d6ba2bf0-7e5f-42cc-8993-5bb616560e1b');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = px.box(\n",
    "    x = df[\"Oldpeak\"], \n",
    "    title= \"Oldpeak Distribution\",\n",
    "    template=\"plotly_white\",\n",
    "    labels={\"x\": \"Oldpeak\"}\n",
    ")\n",
    "\n",
    "custome_layout()\n",
    "\n",
    "\n",
    "iplot(fig)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "dcb3bb11",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:51.197797Z",
     "iopub.status.busy": "2024-03-10T10:04:51.197004Z",
     "iopub.status.idle": "2024-03-10T10:04:51.355985Z",
     "shell.execute_reply": "2024-03-10T10:04:51.354456Z"
    },
    "papermill": {
     "duration": 0.209486,
     "end_time": "2024-03-10T10:04:51.359248",
     "exception": false,
     "start_time": "2024-03-10T10:04:51.149762",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>                            <div id=\"a3a156fb-8f30-4d74-a4db-81a497738758\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"a3a156fb-8f30-4d74-a4db-81a497738758\")) {                    Plotly.newPlot(                        \"a3a156fb-8f30-4d74-a4db-81a497738758\",                        [{\"domain\":{\"x\":[0.0,1.0],\"y\":[0.0,1.0]},\"hovertemplate\":\"label=%{label}\\u003cbr\\u003evalue=%{value}\\u003cextra\\u003e\\u003c\\u002fextra\\u003e\",\"labels\":[\"Heart Disease\",\"No Heart Disease\"],\"legendgroup\":\"\",\"marker\":{\"line\":{\"color\":\"#222\",\"width\":2}},\"name\":\"\",\"pull\":[0,0.09],\"showlegend\":true,\"textfont\":{\"color\":\"#fff\",\"family\":\"arial\",\"size\":16},\"textinfo\":\"value\",\"values\":[14,1],\"type\":\"pie\"}],                        {\"hoverlabel\":{\"bgcolor\":\"#111\",\"font\":{\"family\":\"arial\",\"size\":16}},\"legend\":{\"tracegroupgap\":0},\"showlegend\":true,\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"white\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"#C8D4E3\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"white\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"radialaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"yaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"zaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"caxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2}}},\"title\":{\"font\":{\"family\":\"tahoma\",\"size\":20},\"text\":\"Frequency of Heart Disease With Oldpeak \\u003e 3.7\"}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('a3a156fb-8f30-4d74-a4db-81a497738758');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "high_oldpeak = df[df[\"Oldpeak\"] > 3.7][\"HeartDisease\"].value_counts()\n",
    "\n",
    "fig = px.pie(\n",
    "    data_frame=high_oldpeak,\n",
    "    names = [\"Heart Disease\" if i ==1 else \"No Heart Disease\" for i in high_oldpeak.index],\n",
    "    values = high_oldpeak,\n",
    "    template=\"plotly_white\",\n",
    "    title=\"Frequency of Heart Disease With Oldpeak > 3.7\"\n",
    "\n",
    ")\n",
    "\n",
    "custome_layout(title_size=20, showlegend=True)\n",
    "fig.update_traces(\n",
    "    textinfo=\"value\",\n",
    "    textfont = {\n",
    "        \"size\" : 16,\n",
    "        \"family\" :\"arial\",\n",
    "        \"color\": \"#fff\"\n",
    "    },\n",
    "    marker=dict(line=dict(color='#222', width=2)),\n",
    "    pull= [0,0.09]\n",
    "    \n",
    ")\n",
    "\n",
    "iplot(fig)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "39db5f8a",
   "metadata": {
    "papermill": {
     "duration": 0.044024,
     "end_time": "2024-03-10T10:04:51.447422",
     "exception": false,
     "start_time": "2024-03-10T10:04:51.403398",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<h2 style=\"font: bold 26px tahoma\">\n",
    "    ♠ ST Slope Column \n",
    "</h2>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "257bcf55",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:51.539778Z",
     "iopub.status.busy": "2024-03-10T10:04:51.539330Z",
     "iopub.status.idle": "2024-03-10T10:04:51.551054Z",
     "shell.execute_reply": "2024-03-10T10:04:51.549713Z"
    },
    "papermill": {
     "duration": 0.061898,
     "end_time": "2024-03-10T10:04:51.554010",
     "exception": false,
     "start_time": "2024-03-10T10:04:51.492112",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ST_Slope\n",
       "Flat    50.11%\n",
       "Up      43.03%\n",
       "Down     6.86%\n",
       "Name: proportion, dtype: object"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# ===== To Know What the ST_Slope, Go To The Columns Details 💊 Section =====\n",
    "slope = df[\"ST_Slope\"].value_counts(normalize=1) * 100\n",
    "slope.apply(lambda x: f\"{x:0.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "4fd08504",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:51.646313Z",
     "iopub.status.busy": "2024-03-10T10:04:51.645397Z",
     "iopub.status.idle": "2024-03-10T10:04:51.744152Z",
     "shell.execute_reply": "2024-03-10T10:04:51.742736Z"
    },
    "papermill": {
     "duration": 0.14864,
     "end_time": "2024-03-10T10:04:51.746860",
     "exception": false,
     "start_time": "2024-03-10T10:04:51.598220",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>                            <div id=\"250c7bee-5ddb-4c24-a4ee-a26f3387af8e\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"250c7bee-5ddb-4c24-a4ee-a26f3387af8e\")) {                    Plotly.newPlot(                        \"250c7bee-5ddb-4c24-a4ee-a26f3387af8e\",                        [{\"domain\":{\"x\":[0.0,1.0],\"y\":[0.0,1.0]},\"hole\":0.4,\"hovertemplate\":\"label=%{label}\\u003cbr\\u003evalue=%{value}\\u003cextra\\u003e\\u003c\\u002fextra\\u003e\",\"labels\":[\"Flat\",\"Up\",\"Down\"],\"legendgroup\":\"\",\"marker\":{\"line\":{\"color\":\"#222\",\"width\":2}},\"name\":\"\",\"pull\":[0,0,0.09],\"showlegend\":true,\"textfont\":{\"color\":\"#fff\",\"family\":\"arial\",\"size\":16},\"values\":[50.108932461873636,43.028322440087145,6.862745098039216],\"type\":\"pie\"}],                        {\"hoverlabel\":{\"bgcolor\":\"#111\",\"font\":{\"family\":\"arial\",\"size\":16}},\"legend\":{\"tracegroupgap\":0},\"showlegend\":true,\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"white\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"#C8D4E3\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"white\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"radialaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"yaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"zaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"caxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2}}},\"title\":{\"font\":{\"family\":\"tahoma\",\"size\":20},\"text\":\"Frequency of Heart Disease With Slope \\u003e 3.7\"}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('250c7bee-5ddb-4c24-a4ee-a26f3387af8e');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "fig = px.pie(\n",
    "    data_frame=high_oldpeak,\n",
    "    names = slope.index,\n",
    "    values = slope,\n",
    "    template=\"plotly_white\",\n",
    "    title=\"Frequency of Heart Disease With Slope > 3.7\",\n",
    "    hole=0.4\n",
    "\n",
    ")\n",
    "custome_layout(title_size=20, showlegend=True)\n",
    "fig.update_traces(\n",
    "    textfont = {\n",
    "        \"size\" : 16,\n",
    "        \"family\" :\"arial\",\n",
    "        \"color\": \"#fff\"\n",
    "    },\n",
    "    marker=dict(line=dict(color='#222', width=2)),\n",
    "    pull= [0,0,0.09]\n",
    "    \n",
    ")\n",
    "\n",
    "iplot(fig)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ceaffe33",
   "metadata": {
    "papermill": {
     "duration": 0.045225,
     "end_time": "2024-03-10T10:04:51.838384",
     "exception": false,
     "start_time": "2024-03-10T10:04:51.793159",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<h2 style=\"font: bold 26px tahoma\">\n",
    "    ♠ ST Slope Column \n",
    "</h2>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "2c65048f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:51.931581Z",
     "iopub.status.busy": "2024-03-10T10:04:51.931171Z",
     "iopub.status.idle": "2024-03-10T10:04:51.941165Z",
     "shell.execute_reply": "2024-03-10T10:04:51.939772Z"
    },
    "papermill": {
     "duration": 0.0591,
     "end_time": "2024-03-10T10:04:51.943570",
     "exception": false,
     "start_time": "2024-03-10T10:04:51.884470",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "HeartDisease\n",
       "1    55.337691\n",
       "0    44.662309\n",
       "Name: proportion, dtype: float64"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "target = df[\"HeartDisease\"].value_counts(normalize=1) * 100\n",
    "target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "5dec60b0",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:52.036486Z",
     "iopub.status.busy": "2024-03-10T10:04:52.035560Z",
     "iopub.status.idle": "2024-03-10T10:04:52.151431Z",
     "shell.execute_reply": "2024-03-10T10:04:52.149989Z"
    },
    "papermill": {
     "duration": 0.166192,
     "end_time": "2024-03-10T10:04:52.154177",
     "exception": false,
     "start_time": "2024-03-10T10:04:51.987985",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>                            <div id=\"1a791716-8dd2-47e0-b510-1edfb02e8c1f\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"1a791716-8dd2-47e0-b510-1edfb02e8c1f\")) {                    Plotly.newPlot(                        \"1a791716-8dd2-47e0-b510-1edfb02e8c1f\",                        [{\"alignmentgroup\":\"True\",\"hovertemplate\":\"%{x}\\u003cbr\\u003eFrequency Percentaeg: %{y:0.2f}%\",\"legendgroup\":\"1\",\"marker\":{\"color\":\"#636efa\",\"pattern\":{\"shape\":\"\"}},\"name\":\"1\",\"offsetgroup\":\"1\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"55%\"],\"textfont\":{\"color\":\"#fff\",\"family\":\"consolas\",\"size\":18},\"textposition\":\"auto\",\"x\":[\"Heart Disease\"],\"xaxis\":\"x\",\"y\":[55.33769063180828],\"yaxis\":\"y\",\"type\":\"bar\"},{\"alignmentgroup\":\"True\",\"hovertemplate\":\"%{x}\\u003cbr\\u003eFrequency Percentaeg: %{y:0.2f}%\",\"legendgroup\":\"0\",\"marker\":{\"color\":\"#EF553B\",\"pattern\":{\"shape\":\"\"}},\"name\":\"0\",\"offsetgroup\":\"0\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"45%\"],\"textfont\":{\"color\":\"#fff\",\"family\":\"consolas\",\"size\":18},\"textposition\":\"auto\",\"x\":[\"No Heart Disease\"],\"xaxis\":\"x\",\"y\":[44.66230936819172],\"yaxis\":\"y\",\"type\":\"bar\"}],                        {\"barmode\":\"relative\",\"hoverlabel\":{\"bgcolor\":\"#111\",\"font\":{\"family\":\"arial\",\"size\":16}},\"legend\":{\"title\":{\"text\":\"color\"},\"tracegroupgap\":0},\"showlegend\":false,\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"white\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"#C8D4E3\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"white\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"radialaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"yaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"zaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"caxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2}}},\"title\":{\"font\":{\"family\":\"tahoma\",\"size\":28},\"text\":\"Heart Disease Frequency in Percentage\"},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Target\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Frequency in PCT(%)\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('1a791716-8dd2-47e0-b510-1edfb02e8c1f');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = count_bar_plot(\n",
    "    data_frame=df, column_name=\"HeartDisease\", x_title= \"Target\", y_title=\"Frequency in PCT(%)\", \n",
    "    title=\"Heart Disease Frequency in Percentage\",\n",
    "    hover_template=\"%{x}<br>Frequency Percentaeg: %{y:0.2f}%\",\n",
    "    bars_names=[\"Heart Disease\" if i == 1 else \"No Heart Disease\" for i in target.index]\n",
    ")\n",
    "iplot(fig)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "75ab296d",
   "metadata": {
    "papermill": {
     "duration": 0.044709,
     "end_time": "2024-03-10T10:04:52.243796",
     "exception": false,
     "start_time": "2024-03-10T10:04:52.199087",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Time to Our Lovely Part Ask a Questions and Get Insights😍😍"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1c01cb75",
   "metadata": {
    "papermill": {
     "duration": 0.045226,
     "end_time": "2024-03-10T10:04:52.335622",
     "exception": false,
     "start_time": "2024-03-10T10:04:52.290396",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<p style = \"color: #0B666A;\n",
    "            font: bold 22px tahoma;\n",
    "            padding: 16px;\n",
    "            background-color: #fff;\n",
    "            border: 5px solid #0B666A;\n",
    "            border-radius: 8px\">\n",
    "    ♣ What is the ratio of male to female patients with heart disease ??!\n",
    "</p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "ac244d26",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:52.429170Z",
     "iopub.status.busy": "2024-03-10T10:04:52.427794Z",
     "iopub.status.idle": "2024-03-10T10:04:52.454090Z",
     "shell.execute_reply": "2024-03-10T10:04:52.452941Z"
    },
    "papermill": {
     "duration": 0.075358,
     "end_time": "2024-03-10T10:04:52.456892",
     "exception": false,
     "start_time": "2024-03-10T10:04:52.381534",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Sex</th>\n",
       "      <th>HeartDisease</th>\n",
       "      <th>proportion</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>M</td>\n",
       "      <td>No Heart Disease</td>\n",
       "      <td>36.83</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>M</td>\n",
       "      <td>Heart Disease</td>\n",
       "      <td>63.17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>F</td>\n",
       "      <td>Heart Disease</td>\n",
       "      <td>25.91</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>F</td>\n",
       "      <td>No Heart Disease</td>\n",
       "      <td>74.09</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Sex      HeartDisease  proportion\n",
       "0   M  No Heart Disease       36.83\n",
       "1   M     Heart Disease       63.17\n",
       "2   F     Heart Disease       25.91\n",
       "3   F  No Heart Disease       74.09"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "groupped = df.groupby(\"Sex\", as_index=False)[\"HeartDisease\"].value_counts(normalize=1)\n",
    "groupped[\"HeartDisease\"] = groupped[\"HeartDisease\"].map(lambda x: \"Heart Disease\" if x == 1 else \"No Heart Disease\")\n",
    "groupped[\"proportion\"] = np.round(groupped[\"proportion\"] *100, 2)\n",
    "groupped.sort_index(ascending=False, inplace=True)\n",
    "groupped.reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "31fe1334",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:52.552325Z",
     "iopub.status.busy": "2024-03-10T10:04:52.551891Z",
     "iopub.status.idle": "2024-03-10T10:04:52.695748Z",
     "shell.execute_reply": "2024-03-10T10:04:52.694129Z"
    },
    "papermill": {
     "duration": 0.194297,
     "end_time": "2024-03-10T10:04:52.698873",
     "exception": false,
     "start_time": "2024-03-10T10:04:52.504576",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>                            <div id=\"3c797ba8-edb6-44a3-ac85-ded49d07eab1\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"3c797ba8-edb6-44a3-ac85-ded49d07eab1\")) {                    Plotly.newPlot(                        \"3c797ba8-edb6-44a3-ac85-ded49d07eab1\",                        [{\"alignmentgroup\":\"True\",\"hovertemplate\":\"Gender: %{x}\\u003cbr\\u003ePercentage: %{y:0.2f}%\",\"legendgroup\":\"No Heart Disease\",\"marker\":{\"color\":\"#636efa\",\"line\":{\"color\":\"#222\",\"width\":2},\"pattern\":{\"shape\":\"\"}},\"name\":\"No Heart Disease\",\"offsetgroup\":\"No Heart Disease\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"37%\",\"74%\"],\"textfont\":{\"color\":\"#fff\",\"family\":\"arial\",\"size\":16},\"textposition\":\"auto\",\"x\":[\"Male\",\"Female\"],\"xaxis\":\"x\",\"y\":[36.83,74.09],\"yaxis\":\"y\",\"type\":\"bar\"},{\"alignmentgroup\":\"True\",\"hovertemplate\":\"Gender: %{x}\\u003cbr\\u003ePercentage: %{y:0.2f}%\",\"legendgroup\":\"Heart Disease\",\"marker\":{\"color\":\"#EF553B\",\"line\":{\"color\":\"#222\",\"width\":2},\"pattern\":{\"shape\":\"\"}},\"name\":\"Heart Disease\",\"offsetgroup\":\"Heart Disease\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"63%\",\"26%\"],\"textfont\":{\"color\":\"#fff\",\"family\":\"arial\",\"size\":16},\"textposition\":\"auto\",\"x\":[\"Male\",\"Female\"],\"xaxis\":\"x\",\"y\":[63.17,25.91],\"yaxis\":\"y\",\"type\":\"bar\"}],                        {\"barmode\":\"group\",\"hoverlabel\":{\"bgcolor\":\"#111\",\"font\":{\"family\":\"arial\",\"size\":16}},\"legend\":{\"title\":{\"text\":\"HeartDisease\"},\"tracegroupgap\":0},\"margin\":{\"t\":60},\"showlegend\":true,\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"white\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"#C8D4E3\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"white\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"radialaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"yaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"zaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"caxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2}}},\"title\":{\"font\":{\"family\":\"tahoma\",\"size\":20}},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Gender\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Frequency in PCT(%)\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('3c797ba8-edb6-44a3-ac85-ded49d07eab1');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = px.bar(\n",
    "    data_frame=groupped, \n",
    "    barmode=\"group\", \n",
    "    x = [\"Male\" if i == \"M\" else \"Female\" for i in groupped[\"Sex\"]],\n",
    "    y=\"proportion\", \n",
    "    color=\"HeartDisease\",\n",
    "    template=\"plotly_white\",\n",
    "    labels={\"x\" :\"Gender\", \"proportion\": \"Frequency in PCT(%)\"},\n",
    "    text=groupped[\"proportion\"].apply(lambda x: f\"{x:0.0f}%\"),\n",
    ")\n",
    "\n",
    "custome_layout(title_size=20, showlegend=True)\n",
    "fig.update_traces(\n",
    "    textfont = {\n",
    "        \"size\" : 16,\n",
    "        \"family\" :\"arial\",\n",
    "        \"color\": \"#fff\"\n",
    "    },\n",
    "    marker=dict(line=dict(color='#222', width=2)),\n",
    "    hovertemplate = \"Gender: %{x}<br>Percentage: %{y:0.2f}%\"\n",
    "    \n",
    ")\n",
    "\n",
    "iplot(fig)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "36b33f0a",
   "metadata": {
    "papermill": {
     "duration": 0.046778,
     "end_time": "2024-03-10T10:04:52.794301",
     "exception": false,
     "start_time": "2024-03-10T10:04:52.747523",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<p style = \"color: #F9F07A;\n",
    "            font: bold 18px arial;\n",
    "            background-color: #112;\n",
    "            padding: 20px;\n",
    "            border: 3px solid #FFF67E;\n",
    "            border-radius: 8px\"> \n",
    "    • In case you're unaware, men are typically more susceptible to heart disease than women who have not reached menopause.\n",
    "</p>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d7d1d4b0",
   "metadata": {
    "papermill": {
     "duration": 0.045307,
     "end_time": "2024-03-10T10:04:52.887130",
     "exception": false,
     "start_time": "2024-03-10T10:04:52.841823",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<p style = \"color: #0B666A;\n",
    "            font: bold 22px tahoma;\n",
    "            padding: 16px;\n",
    "            background-color: #fff;\n",
    "            border: 5px solid #0B666A;\n",
    "            border-radius: 8px\">\n",
    "    ♣ What is the frequency of \"chest pain types\" among patients with heart disease??!\n",
    "</p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "9fd03fe5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:52.983221Z",
     "iopub.status.busy": "2024-03-10T10:04:52.982713Z",
     "iopub.status.idle": "2024-03-10T10:04:52.997060Z",
     "shell.execute_reply": "2024-03-10T10:04:52.995361Z"
    },
    "papermill": {
     "duration": 0.065394,
     "end_time": "2024-03-10T10:04:53.000347",
     "exception": false,
     "start_time": "2024-03-10T10:04:52.934953",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ChestPainType\n",
       "ASY    77.165354\n",
       "NAP    14.173228\n",
       "ATA     4.724409\n",
       "TA      3.937008\n",
       "Name: proportion, dtype: float64"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f = df[\"HeartDisease\"] == 1\n",
    "dff = df[f].copy()\n",
    "chest_pain_with_disease = dff[\"ChestPainType\"].value_counts(normalize=1)*100\n",
    "chest_pain_with_disease"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "34a74cbc",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:53.098581Z",
     "iopub.status.busy": "2024-03-10T10:04:53.098131Z",
     "iopub.status.idle": "2024-03-10T10:04:53.249388Z",
     "shell.execute_reply": "2024-03-10T10:04:53.247890Z"
    },
    "papermill": {
     "duration": 0.205137,
     "end_time": "2024-03-10T10:04:53.252349",
     "exception": false,
     "start_time": "2024-03-10T10:04:53.047212",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>                            <div id=\"7813a930-6657-4df3-a86f-b1c09ee3830a\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"7813a930-6657-4df3-a86f-b1c09ee3830a\")) {                    Plotly.newPlot(                        \"7813a930-6657-4df3-a86f-b1c09ee3830a\",                        [{\"alignmentgroup\":\"True\",\"hovertemplate\":\"Chest Pain Type: %{x}\\u003cbr\\u003ePercentage: %{y:0.2f}%\",\"legendgroup\":\"ASY\",\"marker\":{\"color\":\"#636efa\",\"line\":{\"color\":\"#222\",\"width\":2},\"pattern\":{\"shape\":\"\"}},\"name\":\"ASY\",\"offsetgroup\":\"ASY\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"77.2%\"],\"textfont\":{\"family\":\"consolas\",\"size\":16},\"textposition\":\"auto\",\"x\":[\"ASY\"],\"xaxis\":\"x\",\"y\":[77.16535433070865],\"yaxis\":\"y\",\"type\":\"bar\"},{\"alignmentgroup\":\"True\",\"hovertemplate\":\"Chest Pain Type: %{x}\\u003cbr\\u003ePercentage: %{y:0.2f}%\",\"legendgroup\":\"NAP\",\"marker\":{\"color\":\"#EF553B\",\"line\":{\"color\":\"#222\",\"width\":2},\"pattern\":{\"shape\":\"\"}},\"name\":\"NAP\",\"offsetgroup\":\"NAP\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"14.2%\"],\"textfont\":{\"family\":\"consolas\",\"size\":16},\"textposition\":\"auto\",\"x\":[\"NAP\"],\"xaxis\":\"x\",\"y\":[14.173228346456693],\"yaxis\":\"y\",\"type\":\"bar\"},{\"alignmentgroup\":\"True\",\"hovertemplate\":\"Chest Pain Type: %{x}\\u003cbr\\u003ePercentage: %{y:0.2f}%\",\"legendgroup\":\"ATA\",\"marker\":{\"color\":\"#00cc96\",\"line\":{\"color\":\"#222\",\"width\":2},\"pattern\":{\"shape\":\"\"}},\"name\":\"ATA\",\"offsetgroup\":\"ATA\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"4.7%\"],\"textfont\":{\"family\":\"consolas\",\"size\":16},\"textposition\":\"auto\",\"x\":[\"ATA\"],\"xaxis\":\"x\",\"y\":[4.724409448818897],\"yaxis\":\"y\",\"type\":\"bar\"},{\"alignmentgroup\":\"True\",\"hovertemplate\":\"Chest Pain Type: %{x}\\u003cbr\\u003ePercentage: %{y:0.2f}%\",\"legendgroup\":\"TA\",\"marker\":{\"color\":\"#ab63fa\",\"line\":{\"color\":\"#222\",\"width\":2},\"pattern\":{\"shape\":\"\"}},\"name\":\"TA\",\"offsetgroup\":\"TA\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"3.9%\"],\"textfont\":{\"family\":\"consolas\",\"size\":16},\"textposition\":\"auto\",\"x\":[\"TA\"],\"xaxis\":\"x\",\"y\":[3.937007874015748],\"yaxis\":\"y\",\"type\":\"bar\"}],                        {\"barmode\":\"relative\",\"hoverlabel\":{\"bgcolor\":\"#111\",\"font\":{\"family\":\"arial\",\"size\":16}},\"legend\":{\"title\":{\"text\":\"ChestPainType\"},\"tracegroupgap\":0},\"showlegend\":true,\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"white\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"#C8D4E3\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"white\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"radialaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"yaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"zaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"caxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2}}},\"title\":{\"font\":{\"family\":\"tahoma\",\"size\":25},\"text\":\"Chest Pain Type Via Heart patients\"},\"xaxis\":{\"anchor\":\"y\",\"categoryarray\":[\"ASY\",\"NAP\",\"ATA\",\"TA\"],\"categoryorder\":\"array\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"ChestPainType\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Frequency in PCT(%)\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('7813a930-6657-4df3-a86f-b1c09ee3830a');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = px.bar(\n",
    "    data_frame=chest_pain_with_disease, \n",
    "    x = chest_pain_with_disease.index,\n",
    "    y=chest_pain_with_disease, \n",
    "    color=chest_pain_with_disease.index,\n",
    "    template=\"plotly_white\",\n",
    "    labels={\"x\" :\"ChestPainType\", \"y\": \"Frequency in PCT(%)\"},\n",
    "    text=chest_pain_with_disease.apply(lambda x: f\"{x:0.1f}%\"),\n",
    "    title = \"Chest Pain Type Via Heart patients\"\n",
    ")\n",
    "\n",
    "custome_layout(title_size=25, showlegend=True)\n",
    "fig.update_traces(\n",
    "    textfont = {\n",
    "        \"size\" : 16,\n",
    "        \"family\" :\"consolas\",\n",
    "    },\n",
    "    marker=dict(line=dict(color='#222', width=2)),\n",
    "    hovertemplate = \"Chest Pain Type: %{x}<br>Percentage: %{y:0.2f}%\"\n",
    "    \n",
    ")\n",
    "\n",
    "iplot(fig)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0a9f1714",
   "metadata": {
    "papermill": {
     "duration": 0.047071,
     "end_time": "2024-03-10T10:04:53.346155",
     "exception": false,
     "start_time": "2024-03-10T10:04:53.299084",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<p style = \"color: #F9F07A;\n",
    "            font: bold 18px arial;\n",
    "            background-color: #112;\n",
    "            padding: 20px;\n",
    "            border: 3px solid #FFF67E;\n",
    "            border-radius: 8px\"> \n",
    "    • 77% of people with heart disease, patients are \"asymptomatic\" → means they don't have any noticeable symptoms of a particular health issue or disease\n",
    "</p>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2db84660",
   "metadata": {
    "papermill": {
     "duration": 0.04713,
     "end_time": "2024-03-10T10:04:53.441532",
     "exception": false,
     "start_time": "2024-03-10T10:04:53.394402",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<p style = \"color: #0B666A;\n",
    "            font: bold 20px tahoma;\n",
    "            padding: 16px;\n",
    "            background-color: #fff;\n",
    "            border: 5px solid #0B666A;\n",
    "            border-radius: 8px\">\n",
    "    ♣ For those who have heart disease, what is the frequency of patients with exercise angina??!\n",
    "</p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "3ba0c820",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:53.539517Z",
     "iopub.status.busy": "2024-03-10T10:04:53.538555Z",
     "iopub.status.idle": "2024-03-10T10:04:53.551110Z",
     "shell.execute_reply": "2024-03-10T10:04:53.550003Z"
    },
    "papermill": {
     "duration": 0.064916,
     "end_time": "2024-03-10T10:04:53.554078",
     "exception": false,
     "start_time": "2024-03-10T10:04:53.489162",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ExerciseAngina\n",
       "Y    62.204724\n",
       "N    37.795276\n",
       "Name: proportion, dtype: float64"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f = df[\"HeartDisease\"] == 1\n",
    "dff = df[f].copy()\n",
    "angina_with_disease = dff[\"ExerciseAngina\"].value_counts(normalize=1)*100\n",
    "angina_with_disease"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "40d54478",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:53.645634Z",
     "iopub.status.busy": "2024-03-10T10:04:53.645224Z",
     "iopub.status.idle": "2024-03-10T10:04:53.795958Z",
     "shell.execute_reply": "2024-03-10T10:04:53.794511Z"
    },
    "papermill": {
     "duration": 0.199834,
     "end_time": "2024-03-10T10:04:53.798763",
     "exception": false,
     "start_time": "2024-03-10T10:04:53.598929",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>                            <div id=\"a9297b93-66e1-49e1-8450-10006b5fc6e8\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"a9297b93-66e1-49e1-8450-10006b5fc6e8\")) {                    Plotly.newPlot(                        \"a9297b93-66e1-49e1-8450-10006b5fc6e8\",                        [{\"alignmentgroup\":\"True\",\"hovertemplate\":\"ExerciseAngina: %{x}\\u003cbr\\u003ePercentage: %{y:0.2f}%\",\"legendgroup\":\"Y\",\"marker\":{\"color\":\"#636efa\",\"line\":{\"color\":\"#222\",\"width\":2},\"pattern\":{\"shape\":\"\"}},\"name\":\"Y\",\"offsetgroup\":\"Y\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"62.2%\"],\"textfont\":{\"color\":\"#fff\",\"family\":\"consolas\",\"size\":17},\"textposition\":\"auto\",\"x\":[\"Yes\"],\"xaxis\":\"x\",\"y\":[62.20472440944882],\"yaxis\":\"y\",\"type\":\"bar\"},{\"alignmentgroup\":\"True\",\"hovertemplate\":\"ExerciseAngina: %{x}\\u003cbr\\u003ePercentage: %{y:0.2f}%\",\"legendgroup\":\"N\",\"marker\":{\"color\":\"#EF553B\",\"line\":{\"color\":\"#222\",\"width\":2},\"pattern\":{\"shape\":\"\"}},\"name\":\"N\",\"offsetgroup\":\"N\",\"orientation\":\"v\",\"showlegend\":true,\"text\":[\"37.8%\"],\"textfont\":{\"color\":\"#fff\",\"family\":\"consolas\",\"size\":17},\"textposition\":\"auto\",\"x\":[\"No\"],\"xaxis\":\"x\",\"y\":[37.79527559055118],\"yaxis\":\"y\",\"type\":\"bar\"}],                        {\"barmode\":\"relative\",\"hoverlabel\":{\"bgcolor\":\"#111\",\"font\":{\"family\":\"arial\",\"size\":16}},\"legend\":{\"title\":{\"text\":\"ExerciseAngina\"},\"tracegroupgap\":0},\"showlegend\":false,\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"white\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"#C8D4E3\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"white\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"radialaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"yaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"zaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"caxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2}}},\"title\":{\"font\":{\"family\":\"tahoma\",\"size\":25},\"text\":\"Exercise Angina Via Heart Patients\"},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"ExerciseAngina\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Frequency in PCT(%)\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('a9297b93-66e1-49e1-8450-10006b5fc6e8');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = px.bar(\n",
    "    data_frame=angina_with_disease, \n",
    "    x = [\"Yes\" if i == \"Y\" else \"No\" for i in angina_with_disease.index],\n",
    "    y=angina_with_disease, \n",
    "    color=angina_with_disease.index,\n",
    "    template=\"plotly_white\",\n",
    "    labels={\"x\" :\"ExerciseAngina\", \"y\": \"Frequency in PCT(%)\"},\n",
    "    text=angina_with_disease.apply(lambda x: f\"{x:0.1f}%\"),\n",
    "    title= \"Exercise Angina Via Heart Patients\"\n",
    ")\n",
    "\n",
    "custome_layout(title_size=25, showlegend=False)\n",
    "fig.update_traces(\n",
    "    textfont = {\n",
    "        \"size\" : 17,\n",
    "        \"family\" :\"consolas\",\n",
    "        \"color\": \"#fff\"\n",
    "    },\n",
    "    marker=dict(line=dict(color='#222', width=2)),\n",
    "    hovertemplate = \"ExerciseAngina: %{x}<br>Percentage: %{y:0.2f}%\"\n",
    "    \n",
    ")\n",
    "\n",
    "iplot(fig)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b900102e",
   "metadata": {
    "papermill": {
     "duration": 0.048148,
     "end_time": "2024-03-10T10:04:53.895017",
     "exception": false,
     "start_time": "2024-03-10T10:04:53.846869",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Correlations Between Data 📊"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "f0e27802",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:53.993345Z",
     "iopub.status.busy": "2024-03-10T10:04:53.992546Z",
     "iopub.status.idle": "2024-03-10T10:04:54.127670Z",
     "shell.execute_reply": "2024-03-10T10:04:54.126365Z"
    },
    "papermill": {
     "duration": 0.187362,
     "end_time": "2024-03-10T10:04:54.130611",
     "exception": false,
     "start_time": "2024-03-10T10:04:53.943249",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>                            <div id=\"a498bab9-6b99-4476-8bb7-931a78fb8ea3\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"a498bab9-6b99-4476-8bb7-931a78fb8ea3\")) {                    Plotly.newPlot(                        \"a498bab9-6b99-4476-8bb7-931a78fb8ea3\",                        [{\"coloraxis\":\"coloraxis\",\"hovertemplate\":\"x: %{x}\\u003cbr\\u003ey: %{y}\\u003cbr\\u003ecolor: %{z}\\u003cextra\\u003e\\u003c\\u002fextra\\u003e\",\"name\":\"0\",\"textfont\":{\"family\":\"consolas\",\"size\":15},\"texttemplate\":\"%{z:0.3f}\",\"x\":[\"Age\",\"RestingBP\",\"Cholesterol\",\"FastingBS\",\"MaxHR\",\"Oldpeak\",\"HeartDisease\"],\"xaxis\":\"x\",\"y\":[\"Age\",\"RestingBP\",\"Cholesterol\",\"FastingBS\",\"MaxHR\",\"Oldpeak\",\"HeartDisease\"],\"yaxis\":\"y\",\"z\":[[1.0,0.2543993561515428,-0.09528177118121824,0.19803906586674333,-0.3820446750319701,0.25861153601875636,0.2820385058189964],[0.2543993561515428,1.0,0.10089294207709164,0.07019333570992349,-0.11213499711298038,0.16480304317138791,0.10758898037140385],[-0.09528177118121824,0.10089294207709164,1.0,-0.2609743277657631,0.23579240300238535,0.050148109140803906,-0.2327406389270114],[0.19803906586674333,0.07019333570992349,-0.2609743277657631,1.0,-0.1314384913934405,0.05269786028732148,0.26729118611029784],[-0.3820446750319701,-0.11213499711298038,0.23579240300238535,-0.1314384913934405,1.0,-0.1606905500499244,-0.4004207694631906],[0.25861153601875636,0.16480304317138791,0.050148109140803906,0.05269786028732148,-0.1606905500499244,1.0,0.40395072206288607],[0.2820385058189964,0.10758898037140385,-0.2327406389270114,0.26729118611029784,-0.4004207694631906,0.40395072206288607,1.0]],\"type\":\"heatmap\"}],                        {\"coloraxis\":{\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"hoverlabel\":{\"bgcolor\":\"#111\",\"font\":{\"family\":\"arial\",\"size\":16}},\"margin\":{\"t\":60},\"showlegend\":false,\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"white\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"#C8D4E3\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"white\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"radialaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"yaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"zaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"caxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2}}},\"title\":{\"font\":{\"family\":\"tahoma\",\"size\":30}},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0]},\"yaxis\":{\"anchor\":\"x\",\"autorange\":\"reversed\",\"domain\":[0.0,1.0]}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('a498bab9-6b99-4476-8bb7-931a78fb8ea3');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "correlations = df.corr(numeric_only=True)\n",
    "fig = px.imshow(correlations, template='plotly_white', aspect=True, text_auto=\"0.3f\")\n",
    "\n",
    "custome_layout(title_size=30, showlegend=False)\n",
    "fig.update_traces(\n",
    "    textfont = {\n",
    "        \"size\" : 15,\n",
    "        \"family\" :\"consolas\",\n",
    "    },\n",
    "    \n",
    ")\n",
    "\n",
    "iplot(fig)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "35623a55",
   "metadata": {
    "papermill": {
     "duration": 0.045004,
     "end_time": "2024-03-10T10:04:54.222355",
     "exception": false,
     "start_time": "2024-03-10T10:04:54.177351",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<p style = \"color: #0B666A;\n",
    "            font: bold 20px tahoma;\n",
    "            padding: 16px;\n",
    "            background-color: #fff;\n",
    "            border: 5px solid #0B666A;\n",
    "            border-radius: 8px\">\n",
    "    ♣ Relation Between Heart Disease & Max Heart Rate\n",
    "</p>\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "753302d8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:54.320760Z",
     "iopub.status.busy": "2024-03-10T10:04:54.320372Z",
     "iopub.status.idle": "2024-03-10T10:04:56.563888Z",
     "shell.execute_reply": "2024-03-10T10:04:56.562357Z"
    },
    "papermill": {
     "duration": 2.294456,
     "end_time": "2024-03-10T10:04:56.567020",
     "exception": false,
     "start_time": "2024-03-10T10:04:54.272564",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>                            <div id=\"15ae6f65-3811-4fa3-a932-ec4f603c77ca\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"15ae6f65-3811-4fa3-a932-ec4f603c77ca\")) {                    Plotly.newPlot(                        \"15ae6f65-3811-4fa3-a932-ec4f603c77ca\",                        [{\"hovertemplate\":\"Age=%{x}\\u003cbr\\u003eRestingBP=%{y}\\u003cbr\\u003eHeartDisease=%{marker.color}\\u003cextra\\u003e\\u003c\\u002fextra\\u003e\",\"legendgroup\":\"\",\"marker\":{\"color\":[0,1,0,1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,1,0,1,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,1,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,1,0,0,1,0,1,1,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,1,0,0,0,1,1,1,0,1,1,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,1,1,1,1,1,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,1,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,1,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,1,1,0,0,1,0,1,0,0,0,1,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,0,1,0,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,1,0,1,1,0,1,1,1,1,0,1,1,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,0,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,0,1,0,1,1,0,1,0,1,1,1,0,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,0,0,1,1,1,1,1,0,1,1,0,1,1,1,0,0,1,1,1,1,1,0,1,0,1,1,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,1,1,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,1,1,1,1,0,0,1,0,0,0,1,0,1,1,1,1,1,0,0,0,0,0,1,0,1,1,0,1,0,0,0,1,0,1,0,1,1,0,0,0,0,1,0,0,0,0,1,1,1,0,0,0,0,0,0,1,0,1,1,1,1,1,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,0,1,0,0,0,1,1,0,1,1,1,1,0,0,0,1,0,0,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0,0,1,0,0,1,1,1,0,1,0,0,0,0,0,1,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0,0,1,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,1,1,1,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,0,1,0,0,1,0,0,1,0,1,1,0,1,1,1,0,1,0,0,0,0,1,1,0,0,1,1,0,1,0,0,0,0,1,0,0,1,1,1,0,0,0,1,0,1,0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,0],\"coloraxis\":\"coloraxis\",\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"\",\"orientation\":\"v\",\"showlegend\":false,\"x\":[40,49,37,48,54,39,45,54,37,48,37,58,39,49,42,54,38,43,60,36,43,44,49,44,40,36,53,52,53,51,53,56,54,41,43,32,65,41,48,48,54,54,35,52,43,59,37,50,36,41,50,47,45,41,52,51,31,58,54,52,49,43,45,46,50,37,45,32,52,44,57,44,52,44,55,46,32,35,52,49,55,54,63,52,56,66,65,53,43,55,49,39,52,48,39,58,43,39,56,41,65,51,40,40,46,57,48,34,50,39,59,57,47,38,49,33,38,59,35,34,47,52,46,58,58,54,34,48,54,42,38,46,56,56,61,49,43,39,54,43,52,50,47,53,56,39,42,43,50,54,39,48,40,55,41,56,38,49,44,54,59,49,47,42,52,46,50,48,58,58,29,40,53,49,52,43,54,59,37,46,52,51,52,46,54,58,58,41,50,53,46,50,48,45,41,62,49,42,53,57,47,46,42,31,56,50,35,35,28,54,48,50,56,56,47,30,39,54,55,29,46,51,48,33,55,50,53,38,41,37,37,40,38,41,54,39,41,55,48,48,55,54,55,43,48,54,54,48,45,49,44,48,61,62,55,53,55,36,51,55,46,54,46,59,47,54,52,34,54,47,45,32,55,55,45,59,51,52,57,54,60,49,51,55,42,51,59,53,48,36,48,47,53,65,32,61,50,57,51,47,60,55,53,62,51,51,55,53,58,57,65,60,41,34,53,74,57,56,61,68,59,63,38,62,46,42,45,59,52,60,60,56,38,40,51,62,72,63,63,64,43,64,61,52,51,69,59,48,69,36,53,43,56,58,55,67,46,53,38,53,62,47,56,56,56,64,61,68,57,63,60,66,63,59,61,73,47,65,70,50,60,50,43,38,54,61,42,53,55,61,51,70,61,38,57,38,62,58,52,61,50,51,65,52,47,35,57,62,59,53,62,54,56,56,54,66,63,44,60,55,66,66,65,60,60,60,56,59,62,63,57,62,63,46,63,60,58,64,63,74,52,69,51,60,56,55,54,77,63,55,52,64,60,60,58,59,61,40,61,41,57,63,59,51,59,42,55,63,62,56,53,68,53,60,62,59,51,61,57,56,58,69,67,58,65,63,55,57,65,54,72,75,49,51,60,64,58,61,67,62,65,63,69,51,62,55,75,40,67,58,60,63,35,62,43,63,68,65,48,63,64,61,50,59,55,45,65,61,49,72,50,64,55,63,59,56,62,74,54,57,62,76,54,70,61,48,48,61,66,68,55,62,71,74,53,58,75,56,58,64,54,54,59,55,57,61,41,71,38,55,56,69,64,72,69,56,62,67,57,69,51,48,69,69,64,57,53,37,67,74,63,58,61,64,58,60,57,55,55,56,57,61,61,74,68,51,62,53,62,46,54,62,55,58,62,70,67,57,64,74,65,56,59,60,63,59,53,44,61,57,71,46,53,64,40,67,48,43,47,54,48,46,51,58,71,57,66,37,59,50,48,61,59,42,48,40,62,44,46,59,58,49,44,66,65,42,52,65,63,45,41,61,60,59,62,57,51,44,60,63,57,51,58,44,47,61,57,70,76,67,45,45,39,42,56,58,35,58,41,57,42,62,59,41,50,59,61,54,54,52,47,66,58,64,50,44,67,49,57,63,48,51,60,59,45,55,41,60,54,42,49,46,56,66,56,49,54,57,65,54,54,62,52,52,60,63,66,42,64,54,46,67,56,34,57,64,59,50,51,54,53,52,40,58,41,41,50,54,64,51,46,55,45,56,66,38,62,55,58,43,64,50,53,45,65,69,69,67,68,34,62,51,46,67,50,42,56,41,42,53,43,56,52,62,70,54,70,54,35,48,55,58,54,69,77,68,58,60,51,55,52,60,58,64,37,59,51,43,58,29,41,63,51,54,44,54,65,57,63,35,41,62,43,58,52,61,39,45,52,62,62,53,43,47,52,68,39,53,62,51,60,65,65,60,60,54,44,44,51,59,71,61,55,64,43,58,60,58,49,48,52,44,56,57,67,53,52,43,52,59,64,66,39,57,58,57,47,55,35,61,58,58,58,56,56,67,55,44,63,63,41,59,57,45,68,57,57,38],\"xaxis\":\"x\",\"y\":[140,160,130,138,150,120,130,110,140,120,130,136,120,140,115,120,110,120,100,120,100,120,124,150,130,130,124,120,113,125,145,130,125,130,150,125,140,110,120,150,150,130,150,140,120,130,120,140,112,110,130,120,140,130,130,160,120,130,150,112,100,150,140,120,110,120,132,110,160,150,140,130,120,120,140,150,118,140,140,130,110,120,150,160,150,140,170,140,120,140,110,130,120,160,110,130,142,160,120,125,130,130,150,120,118,140,120,150,140,190,130,150,140,140,130,100,120,130,120,140,135,125,110,180,130,120,130,108,120,120,145,110,170,150,130,115,120,120,140,150,160,140,160,140,120,110,120,120,120,130,130,100,130,120,120,155,110,140,130,160,140,128,160,120,140,140,140,140,135,140,120,140,140,140,140,140,140,140,130,130,130,130,140,110,160,160,130,120,120,180,180,170,130,135,125,160,120,150,120,130,110,120,160,100,130,150,120,110,130,125,106,140,130,130,150,170,110,120,140,140,130,160,120,120,120,145,120,92,120,130,130,130,120,112,140,120,120,140,160,160,145,200,160,120,160,120,120,122,130,130,135,120,125,140,145,120,130,150,150,122,140,120,120,130,140,160,130,98,130,130,120,105,140,120,180,180,135,170,180,130,120,150,130,110,140,110,140,120,133,120,110,140,130,115,95,105,145,110,110,110,160,140,125,120,95,120,115,130,115,95,155,125,125,115,80,145,105,140,130,145,125,100,105,115,100,105,110,125,95,130,115,115,100,95,130,120,160,150,140,95,100,110,110,130,120,135,120,115,137,110,120,140,120,130,120,145,115,120,115,105,160,160,155,120,120,200,150,135,140,150,135,150,185,135,125,160,155,160,140,120,160,115,115,110,120,150,145,130,140,160,140,115,130,150,160,135,140,170,165,200,160,130,145,135,110,120,140,115,110,160,150,180,125,125,130,155,140,130,132,142,110,120,150,180,120,160,126,140,110,133,128,120,170,110,126,152,116,120,130,138,128,130,128,130,120,136,130,124,160,0,122,144,140,120,136,154,120,125,134,104,139,136,122,128,131,134,120,132,152,124,126,138,154,141,131,178,132,110,130,170,126,140,142,120,134,139,110,140,140,136,120,170,130,137,142,142,132,146,160,135,136,130,140,132,158,136,136,106,120,110,136,160,123,112,122,130,150,150,102,96,130,120,144,124,150,130,144,139,131,143,133,143,116,110,125,130,133,150,130,110,138,104,138,170,140,132,132,142,112,139,172,120,144,145,155,150,160,137,137,134,133,132,140,135,144,141,150,130,110,158,128,140,150,160,142,137,139,146,156,145,131,140,122,142,141,180,124,118,140,140,136,100,190,130,160,130,122,133,120,130,130,140,120,155,134,114,160,144,158,134,127,135,122,140,120,130,115,124,128,120,120,130,110,140,150,135,142,140,134,128,112,140,140,110,140,120,130,115,112,132,130,138,120,112,110,128,160,120,170,144,130,140,160,130,122,152,124,130,101,126,140,118,110,160,150,136,128,140,140,130,105,138,120,174,120,150,130,120,150,145,150,140,136,118,108,120,120,156,140,106,142,104,94,120,120,146,120,150,130,110,148,128,178,126,150,140,130,124,110,125,110,120,100,140,120,108,120,130,165,130,124,100,150,140,112,180,110,158,135,120,134,120,200,150,130,120,122,152,160,125,160,120,136,134,117,108,112,140,120,150,142,152,125,118,132,145,138,140,125,192,123,112,110,132,112,112,120,108,130,130,105,140,128,120,178,120,150,130,128,110,180,110,130,138,138,160,140,100,120,118,138,140,150,125,129,120,134,110,102,130,130,132,108,140,160,140,145,108,126,124,135,100,110,140,125,118,125,125,140,160,152,102,105,125,130,170,125,122,128,130,130,135,94,120,120,110,135,150,130,138,135,130,132,150,118,145,118,115,128,130,160,138,120,138,120,180,140,130,140,140,130,110,155,140,145,120,130,112,110,150,160,150,132,140,150,120,130,120,130,110,172,120,140,140,160,128,138,132,128,134,170,146,138,154,130,110,130,128,122,148,114,170,125,130,120,152,132,120,140,124,120,164,140,110,144,130,130,138],\"yaxis\":\"y\",\"type\":\"scatter\"},{\"hovertemplate\":\"\\u003cb\\u003eOLS trendline\\u003c\\u002fb\\u003e\\u003cbr\\u003eRestingBP = 0.49933 * Age + 105.677\\u003cbr\\u003eR\\u003csup\\u003e2\\u003c\\u002fsup\\u003e=0.064719\\u003cbr\\u003e\\u003cbr\\u003eAge=%{x}\\u003cbr\\u003eRestingBP=%{y} \\u003cb\\u003e(trend)\\u003c\\u002fb\\u003e\\u003cextra\\u003e\\u003c\\u002fextra\\u003e\",\"legendgroup\":\"\",\"marker\":{\"symbol\":\"circle\"},\"mode\":\"lines\",\"name\":\"\",\"showlegend\":false,\"x\":[28,29,29,29,30,31,31,32,32,32,32,32,33,33,34,34,34,34,34,34,34,35,35,35,35,35,35,35,35,35,35,35,36,36,36,36,36,36,37,37,37,37,37,37,37,37,37,37,37,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,40,40,40,40,40,40,40,40,40,40,40,40,40,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,66,66,66,66,66,66,66,66,66,66,66,66,66,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,68,68,68,68,68,68,68,68,68,68,69,69,69,69,69,69,69,69,69,69,69,69,69,70,70,70,70,70,70,70,71,71,71,71,71,72,72,72,72,73,74,74,74,74,74,74,74,75,75,75,76,76,77,77],\"xaxis\":\"x\",\"y\":[119.65815851816923,120.15748856985039,120.15748856985039,120.15748856985039,120.65681862153156,121.15614867321271,121.15614867321271,121.65547872489387,121.65547872489387,121.65547872489387,121.65547872489387,121.65547872489387,122.15480877657502,122.15480877657502,122.65413882825618,122.65413882825618,122.65413882825618,122.65413882825618,122.65413882825618,122.65413882825618,122.65413882825618,123.15346887993735,123.15346887993735,123.15346887993735,123.15346887993735,123.15346887993735,123.15346887993735,123.15346887993735,123.15346887993735,123.15346887993735,123.15346887993735,123.15346887993735,123.6527989316185,123.6527989316185,123.6527989316185,123.6527989316185,123.6527989316185,123.6527989316185,124.15212898329966,124.15212898329966,124.15212898329966,124.15212898329966,124.15212898329966,124.15212898329966,124.15212898329966,124.15212898329966,124.15212898329966,124.15212898329966,124.15212898329966,124.65145903498082,124.65145903498082,124.65145903498082,124.65145903498082,124.65145903498082,124.65145903498082,124.65145903498082,124.65145903498082,124.65145903498082,124.65145903498082,124.65145903498082,124.65145903498082,124.65145903498082,124.65145903498082,124.65145903498082,124.65145903498082,125.15078908666197,125.15078908666197,125.15078908666197,125.15078908666197,125.15078908666197,125.15078908666197,125.15078908666197,125.15078908666197,125.15078908666197,125.15078908666197,125.15078908666197,125.15078908666197,125.15078908666197,125.15078908666197,125.15078908666197,125.65011913834314,125.65011913834314,125.65011913834314,125.65011913834314,125.65011913834314,125.65011913834314,125.65011913834314,125.65011913834314,125.65011913834314,125.65011913834314,125.65011913834314,125.65011913834314,125.65011913834314,126.1494491900243,126.1494491900243,126.1494491900243,126.1494491900243,126.1494491900243,126.1494491900243,126.1494491900243,126.1494491900243,126.1494491900243,126.1494491900243,126.1494491900243,126.1494491900243,126.1494491900243,126.1494491900243,126.1494491900243,126.1494491900243,126.1494491900243,126.1494491900243,126.1494491900243,126.1494491900243,126.1494491900243,126.1494491900243,126.1494491900243,126.1494491900243,126.64877924170545,126.64877924170545,126.64877924170545,126.64877924170545,126.64877924170545,126.64877924170545,126.64877924170545,126.64877924170545,126.64877924170545,126.64877924170545,126.64877924170545,126.64877924170545,126.64877924170545,126.64877924170545,126.64877924170545,126.64877924170545,126.64877924170545,126.64877924170545,127.14810929338661,127.14810929338661,127.14810929338661,127.14810929338661,127.14810929338661,127.14810929338661,127.14810929338661,127.14810929338661,127.14810929338661,127.14810929338661,127.14810929338661,127.14810929338661,127.14810929338661,127.14810929338661,127.14810929338661,127.14810929338661,127.14810929338661,127.14810929338661,127.14810929338661,127.14810929338661,127.14810929338661,127.14810929338661,127.14810929338661,127.14810929338661,127.64743934506777,127.64743934506777,127.64743934506777,127.64743934506777,127.64743934506777,127.64743934506777,127.64743934506777,127.64743934506777,127.64743934506777,127.64743934506777,127.64743934506777,127.64743934506777,127.64743934506777,127.64743934506777,127.64743934506777,127.64743934506777,127.64743934506777,127.64743934506777,127.64743934506777,128.14676939674894,128.14676939674894,128.14676939674894,128.14676939674894,128.14676939674894,128.14676939674894,128.14676939674894,128.14676939674894,128.14676939674894,128.14676939674894,128.14676939674894,128.14676939674894,128.14676939674894,128.14676939674894,128.14676939674894,128.14676939674894,128.14676939674894,128.14676939674894,128.6460994484301,128.6460994484301,128.6460994484301,128.6460994484301,128.6460994484301,128.6460994484301,128.6460994484301,128.6460994484301,128.6460994484301,128.6460994484301,128.6460994484301,128.6460994484301,128.6460994484301,128.6460994484301,128.6460994484301,128.6460994484301,128.6460994484301,128.6460994484301,128.6460994484301,128.6460994484301,128.6460994484301,128.6460994484301,128.6460994484301,128.6460994484301,129.14542950011125,129.14542950011125,129.14542950011125,129.14542950011125,129.14542950011125,129.14542950011125,129.14542950011125,129.14542950011125,129.14542950011125,129.14542950011125,129.14542950011125,129.14542950011125,129.14542950011125,129.14542950011125,129.14542950011125,129.14542950011125,129.14542950011125,129.14542950011125,129.14542950011125,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,129.64475955179242,130.14408960347356,130.14408960347356,130.14408960347356,130.14408960347356,130.14408960347356,130.14408960347356,130.14408960347356,130.14408960347356,130.14408960347356,130.14408960347356,130.14408960347356,130.14408960347356,130.14408960347356,130.14408960347356,130.14408960347356,130.14408960347356,130.14408960347356,130.14408960347356,130.14408960347356,130.14408960347356,130.14408960347356,130.64341965515473,130.64341965515473,130.64341965515473,130.64341965515473,130.64341965515473,130.64341965515473,130.64341965515473,130.64341965515473,130.64341965515473,130.64341965515473,130.64341965515473,130.64341965515473,130.64341965515473,130.64341965515473,130.64341965515473,130.64341965515473,130.64341965515473,130.64341965515473,130.64341965515473,130.64341965515473,130.64341965515473,130.64341965515473,130.64341965515473,130.64341965515473,130.64341965515473,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.1427497068359,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,131.64207975851704,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.1414098101982,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,132.64073986187935,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.14006991356052,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,133.6393999652417,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.13873001692284,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,134.638060068604,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.13739012028515,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,135.63672017196632,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.1360502236475,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,136.63538027532863,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.1347103270098,137.63404037869094,137.63404037869094,137.63404037869094,137.63404037869094,137.63404037869094,137.63404037869094,137.63404037869094,137.63404037869094,137.63404037869094,137.63404037869094,137.63404037869094,137.63404037869094,137.63404037869094,137.63404037869094,137.63404037869094,137.63404037869094,137.63404037869094,137.63404037869094,137.63404037869094,137.63404037869094,137.63404037869094,137.63404037869094,138.1333704303721,138.1333704303721,138.1333704303721,138.1333704303721,138.1333704303721,138.1333704303721,138.1333704303721,138.1333704303721,138.1333704303721,138.1333704303721,138.1333704303721,138.1333704303721,138.1333704303721,138.1333704303721,138.1333704303721,138.1333704303721,138.1333704303721,138.1333704303721,138.1333704303721,138.1333704303721,138.1333704303721,138.63270048205328,138.63270048205328,138.63270048205328,138.63270048205328,138.63270048205328,138.63270048205328,138.63270048205328,138.63270048205328,138.63270048205328,138.63270048205328,138.63270048205328,138.63270048205328,138.63270048205328,139.13203053373442,139.13203053373442,139.13203053373442,139.13203053373442,139.13203053373442,139.13203053373442,139.13203053373442,139.13203053373442,139.13203053373442,139.13203053373442,139.13203053373442,139.13203053373442,139.13203053373442,139.13203053373442,139.13203053373442,139.6313605854156,139.6313605854156,139.6313605854156,139.6313605854156,139.6313605854156,139.6313605854156,139.6313605854156,139.6313605854156,139.6313605854156,139.6313605854156,140.13069063709673,140.13069063709673,140.13069063709673,140.13069063709673,140.13069063709673,140.13069063709673,140.13069063709673,140.13069063709673,140.13069063709673,140.13069063709673,140.13069063709673,140.13069063709673,140.13069063709673,140.6300206887779,140.6300206887779,140.6300206887779,140.6300206887779,140.6300206887779,140.6300206887779,140.6300206887779,141.12935074045907,141.12935074045907,141.12935074045907,141.12935074045907,141.12935074045907,141.62868079214022,141.62868079214022,141.62868079214022,141.62868079214022,142.1280108438214,142.62734089550253,142.62734089550253,142.62734089550253,142.62734089550253,142.62734089550253,142.62734089550253,142.62734089550253,143.1266709471837,143.1266709471837,143.1266709471837,143.62600099886487,143.62600099886487,144.125331050546,144.125331050546],\"yaxis\":\"y\",\"type\":\"scatter\"}],                        {\"coloraxis\":{\"colorbar\":{\"title\":{\"text\":\"HeartDisease\"}},\"colorscale\":[[0.0,\"rgb(103,0,31)\"],[0.1,\"rgb(178,24,43)\"],[0.2,\"rgb(214,96,77)\"],[0.3,\"rgb(244,165,130)\"],[0.4,\"rgb(253,219,199)\"],[0.5,\"rgb(247,247,247)\"],[0.6,\"rgb(209,229,240)\"],[0.7,\"rgb(146,197,222)\"],[0.8,\"rgb(67,147,195)\"],[0.9,\"rgb(33,102,172)\"],[1.0,\"rgb(5,48,97)\"]]},\"hoverlabel\":{\"bgcolor\":\"#111\",\"font\":{\"family\":\"arial\",\"size\":16}},\"legend\":{\"tracegroupgap\":0},\"showlegend\":false,\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"white\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"#C8D4E3\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"white\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"radialaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"yaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"zaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"caxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2}}},\"title\":{\"font\":{\"family\":\"tahoma\",\"size\":30},\"text\":\"Blood Pressure Via Age\"},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Age\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"RestingBP\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('15ae6f65-3811-4fa3-a932-ec4f603c77ca');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = px.scatter(\n",
    "    data_frame=df,\n",
    "    x = \"Age\",\n",
    "    y = \"RestingBP\",\n",
    "    trendline=\"ols\",\n",
    "    color = \"HeartDisease\",\n",
    "    template=\"plotly_white\",\n",
    "    color_continuous_scale=\"RdBu\",\n",
    "    title=\"Blood Pressure Via Age\"\n",
    ")\n",
    "custome_layout(title_size=30, showlegend=False)\n",
    "\n",
    "iplot(fig)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5eb8479d",
   "metadata": {
    "papermill": {
     "duration": 0.048414,
     "end_time": "2024-03-10T10:04:56.665054",
     "exception": false,
     "start_time": "2024-03-10T10:04:56.616640",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<p style = \"color: #0B666A;\n",
    "            font: bold 20px tahoma;\n",
    "            padding: 16px;\n",
    "            background-color: #fff;\n",
    "            border: 5px solid #0B666A;\n",
    "            border-radius: 8px\">\n",
    "    ♣ Relation Between Age & Max Heart Rate\n",
    "</p>\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "52b7426b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:56.763426Z",
     "iopub.status.busy": "2024-03-10T10:04:56.762644Z",
     "iopub.status.idle": "2024-03-10T10:04:56.888991Z",
     "shell.execute_reply": "2024-03-10T10:04:56.887739Z"
    },
    "papermill": {
     "duration": 0.179326,
     "end_time": "2024-03-10T10:04:56.892275",
     "exception": false,
     "start_time": "2024-03-10T10:04:56.712949",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>                            <div id=\"ac2b45b2-f8b1-4cff-b680-19598eaa4311\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"ac2b45b2-f8b1-4cff-b680-19598eaa4311\")) {                    Plotly.newPlot(                        \"ac2b45b2-f8b1-4cff-b680-19598eaa4311\",                        [{\"hovertemplate\":\"Age=%{x}\\u003cbr\\u003eMaxHR=%{y}\\u003cbr\\u003eHeartDisease=%{marker.color}\\u003cextra\\u003e\\u003c\\u002fextra\\u003e\",\"legendgroup\":\"\",\"marker\":{\"color\":[0,1,0,1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,1,0,1,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,1,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,1,0,0,1,0,1,1,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,1,0,0,0,1,1,1,0,1,1,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,1,1,1,1,1,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,1,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,1,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,1,1,0,0,1,0,1,0,0,0,1,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,0,1,0,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,1,0,1,1,0,1,1,1,1,0,1,1,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,0,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,0,1,0,1,1,0,1,0,1,1,1,0,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,0,0,1,1,1,1,1,0,1,1,0,1,1,1,0,0,1,1,1,1,1,0,1,0,1,1,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,1,1,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,1,1,1,1,0,0,1,0,0,0,1,0,1,1,1,1,1,0,0,0,0,0,1,0,1,1,0,1,0,0,0,1,0,1,0,1,1,0,0,0,0,1,0,0,0,0,1,1,1,0,0,0,0,0,0,1,0,1,1,1,1,1,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,0,1,0,0,0,1,1,0,1,1,1,1,0,0,0,1,0,0,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0,0,1,0,0,1,1,1,0,1,0,0,0,0,0,1,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0,0,1,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,1,1,1,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,0,1,0,0,1,0,0,1,0,1,1,0,1,1,1,0,1,0,0,0,0,1,1,0,0,1,1,0,1,0,0,0,0,1,0,0,1,1,1,0,0,0,1,0,1,0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,0],\"coloraxis\":\"coloraxis\",\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"\",\"orientation\":\"v\",\"showlegend\":false,\"x\":[40,49,37,48,54,39,45,54,37,48,37,58,39,49,42,54,38,43,60,36,43,44,49,44,40,36,53,52,53,51,53,56,54,41,43,32,65,41,48,48,54,54,35,52,43,59,37,50,36,41,50,47,45,41,52,51,31,58,54,52,49,43,45,46,50,37,45,32,52,44,57,44,52,44,55,46,32,35,52,49,55,54,63,52,56,66,65,53,43,55,49,39,52,48,39,58,43,39,56,41,65,51,40,40,46,57,48,34,50,39,59,57,47,38,49,33,38,59,35,34,47,52,46,58,58,54,34,48,54,42,38,46,56,56,61,49,43,39,54,43,52,50,47,53,56,39,42,43,50,54,39,48,40,55,41,56,38,49,44,54,59,49,47,42,52,46,50,48,58,58,29,40,53,49,52,43,54,59,37,46,52,51,52,46,54,58,58,41,50,53,46,50,48,45,41,62,49,42,53,57,47,46,42,31,56,50,35,35,28,54,48,50,56,56,47,30,39,54,55,29,46,51,48,33,55,50,53,38,41,37,37,40,38,41,54,39,41,55,48,48,55,54,55,43,48,54,54,48,45,49,44,48,61,62,55,53,55,36,51,55,46,54,46,59,47,54,52,34,54,47,45,32,55,55,45,59,51,52,57,54,60,49,51,55,42,51,59,53,48,36,48,47,53,65,32,61,50,57,51,47,60,55,53,62,51,51,55,53,58,57,65,60,41,34,53,74,57,56,61,68,59,63,38,62,46,42,45,59,52,60,60,56,38,40,51,62,72,63,63,64,43,64,61,52,51,69,59,48,69,36,53,43,56,58,55,67,46,53,38,53,62,47,56,56,56,64,61,68,57,63,60,66,63,59,61,73,47,65,70,50,60,50,43,38,54,61,42,53,55,61,51,70,61,38,57,38,62,58,52,61,50,51,65,52,47,35,57,62,59,53,62,54,56,56,54,66,63,44,60,55,66,66,65,60,60,60,56,59,62,63,57,62,63,46,63,60,58,64,63,74,52,69,51,60,56,55,54,77,63,55,52,64,60,60,58,59,61,40,61,41,57,63,59,51,59,42,55,63,62,56,53,68,53,60,62,59,51,61,57,56,58,69,67,58,65,63,55,57,65,54,72,75,49,51,60,64,58,61,67,62,65,63,69,51,62,55,75,40,67,58,60,63,35,62,43,63,68,65,48,63,64,61,50,59,55,45,65,61,49,72,50,64,55,63,59,56,62,74,54,57,62,76,54,70,61,48,48,61,66,68,55,62,71,74,53,58,75,56,58,64,54,54,59,55,57,61,41,71,38,55,56,69,64,72,69,56,62,67,57,69,51,48,69,69,64,57,53,37,67,74,63,58,61,64,58,60,57,55,55,56,57,61,61,74,68,51,62,53,62,46,54,62,55,58,62,70,67,57,64,74,65,56,59,60,63,59,53,44,61,57,71,46,53,64,40,67,48,43,47,54,48,46,51,58,71,57,66,37,59,50,48,61,59,42,48,40,62,44,46,59,58,49,44,66,65,42,52,65,63,45,41,61,60,59,62,57,51,44,60,63,57,51,58,44,47,61,57,70,76,67,45,45,39,42,56,58,35,58,41,57,42,62,59,41,50,59,61,54,54,52,47,66,58,64,50,44,67,49,57,63,48,51,60,59,45,55,41,60,54,42,49,46,56,66,56,49,54,57,65,54,54,62,52,52,60,63,66,42,64,54,46,67,56,34,57,64,59,50,51,54,53,52,40,58,41,41,50,54,64,51,46,55,45,56,66,38,62,55,58,43,64,50,53,45,65,69,69,67,68,34,62,51,46,67,50,42,56,41,42,53,43,56,52,62,70,54,70,54,35,48,55,58,54,69,77,68,58,60,51,55,52,60,58,64,37,59,51,43,58,29,41,63,51,54,44,54,65,57,63,35,41,62,43,58,52,61,39,45,52,62,62,53,43,47,52,68,39,53,62,51,60,65,65,60,60,54,44,44,51,59,71,61,55,64,43,58,60,58,49,48,52,44,56,57,67,53,52,43,52,59,64,66,39,57,58,57,47,55,35,61,58,58,58,56,56,67,55,44,63,63,41,59,57,45,68,57,57,38],\"xaxis\":\"x\",\"y\":[172,156,98,108,122,170,170,142,130,120,142,99,145,140,137,150,166,165,125,160,142,142,164,150,138,178,112,118,127,145,130,114,122,130,154,155,87,142,148,130,130,100,168,170,120,120,168,170,184,170,121,98,122,150,140,170,153,140,134,96,174,175,144,125,145,130,144,184,82,170,145,135,150,115,128,116,130,150,138,170,160,154,115,165,125,94,112,142,155,110,160,140,148,92,180,140,138,160,140,144,115,100,130,152,124,140,110,168,135,106,124,92,125,150,135,150,170,130,185,180,170,139,140,110,150,110,190,175,140,152,130,150,122,124,120,175,175,146,118,130,94,125,158,155,150,132,155,176,160,125,120,100,150,140,160,150,150,130,100,130,119,96,174,150,140,175,140,118,100,160,160,188,162,172,134,135,105,150,150,90,120,150,124,140,130,92,110,138,110,120,120,116,160,110,180,116,132,136,116,98,150,150,146,150,100,140,180,140,185,140,110,140,128,164,98,170,150,137,150,170,112,150,125,185,137,150,140,134,170,184,158,167,129,142,140,160,118,136,99,102,155,142,143,118,103,137,150,150,130,120,135,115,115,152,96,130,150,172,120,155,165,138,115,125,145,175,110,150,91,145,140,165,130,134,180,100,150,126,126,155,135,122,160,160,170,120,140,132,156,180,138,135,148,93,127,110,139,131,92,149,149,150,120,123,126,127,155,120,138,182,154,110,176,154,141,123,148,121,77,136,175,109,166,128,133,128,138,119,82,130,143,82,179,144,170,134,114,154,149,145,122,114,113,120,104,130,115,128,104,125,120,140,100,100,92,125,113,95,128,115,72,124,99,148,97,140,117,120,120,86,63,108,98,115,105,121,118,122,157,156,99,120,145,156,155,105,99,135,83,145,60,92,115,120,98,150,143,105,122,70,110,163,67,128,120,130,100,72,94,122,78,150,103,98,110,90,112,127,140,149,99,120,105,140,141,157,140,117,120,120,148,86,84,125,120,118,124,106,111,116,180,129,125,140,120,124,117,110,105,155,110,122,118,133,123,131,80,165,86,111,118,84,117,107,128,160,125,130,97,161,106,130,140,122,130,120,139,108,148,123,110,118,125,106,112,128,180,144,135,140,102,108,145,127,110,140,69,148,130,130,140,138,140,138,112,131,112,80,150,110,126,88,153,150,120,160,132,120,110,121,128,135,120,117,150,144,113,135,127,109,128,115,102,140,135,122,119,130,112,100,122,120,105,129,120,139,162,100,140,135,73,86,108,116,160,118,112,122,124,102,137,141,154,126,160,115,128,115,105,110,119,109,135,130,112,126,120,110,119,110,130,159,84,126,116,120,122,165,122,94,133,110,150,130,113,140,100,136,127,98,96,123,98,112,151,96,108,128,138,126,154,137,100,135,93,109,160,141,105,121,140,142,142,170,154,161,111,180,145,159,125,120,155,144,178,129,180,181,143,159,139,152,157,165,130,150,138,170,140,126,150,138,125,150,186,181,163,179,156,134,165,126,177,120,114,125,184,157,179,175,168,125,96,143,103,173,142,169,171,150,112,186,152,149,152,140,163,143,116,142,147,148,179,173,178,105,130,111,168,126,178,140,145,163,128,164,169,109,108,168,118,151,156,133,162,175,71,163,124,147,166,143,157,162,138,117,153,161,170,162,162,144,133,114,103,139,116,88,151,152,163,99,169,158,160,169,132,178,96,165,160,172,144,192,168,132,182,163,125,195,95,160,114,173,172,179,158,167,122,149,172,111,170,162,165,182,154,155,130,161,154,159,152,152,174,131,146,125,115,174,106,122,147,163,163,194,150,158,122,173,162,105,147,157,112,160,125,156,156,175,161,122,158,151,162,151,171,141,173,145,178,160,154,131,187,159,166,165,131,202,172,172,154,147,170,126,127,174,132,182,132,97,136,162,190,146,140,185,161,146,145,160,120,156,172,150,182,143,160,142,144,158,148,155,142,113,188,153,123,157,162,137,132,158,171,172,132,160,171,168,162,173,153,148,108,115,169,143,156,162,155,152,152,164,131,143,179,130,174,161,140,146,144,163,169,150,166,144,144,136,182,90,123,132,141,115,174,173],\"yaxis\":\"y\",\"type\":\"scatter\"},{\"hovertemplate\":\"\\u003cb\\u003eOLS trendline\\u003c\\u002fb\\u003e\\u003cbr\\u003eMaxHR = -1.03121 * Age + 191.99\\u003cbr\\u003eR\\u003csup\\u003e2\\u003c\\u002fsup\\u003e=0.145958\\u003cbr\\u003e\\u003cbr\\u003eAge=%{x}\\u003cbr\\u003eMaxHR=%{y} \\u003cb\\u003e(trend)\\u003c\\u002fb\\u003e\\u003cextra\\u003e\\u003c\\u002fextra\\u003e\",\"legendgroup\":\"\",\"marker\":{\"symbol\":\"circle\"},\"mode\":\"lines\",\"name\":\"\",\"showlegend\":false,\"x\":[28,29,29,29,30,31,31,32,32,32,32,32,33,33,34,34,34,34,34,34,34,35,35,35,35,35,35,35,35,35,35,35,36,36,36,36,36,36,37,37,37,37,37,37,37,37,37,37,37,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,40,40,40,40,40,40,40,40,40,40,40,40,40,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,66,66,66,66,66,66,66,66,66,66,66,66,66,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,68,68,68,68,68,68,68,68,68,68,69,69,69,69,69,69,69,69,69,69,69,69,69,70,70,70,70,70,70,70,71,71,71,71,71,72,72,72,72,73,74,74,74,74,74,74,74,75,75,75,76,76,77,77],\"xaxis\":\"x\",\"y\":[163.11639316813358,162.08518564738776,162.08518564738776,162.08518564738776,161.05397812664194,160.0227706058961,160.0227706058961,158.9915630851503,158.9915630851503,158.9915630851503,158.9915630851503,158.9915630851503,157.96035556440447,157.96035556440447,156.92914804365864,156.92914804365864,156.92914804365864,156.92914804365864,156.92914804365864,156.92914804365864,156.92914804365864,155.89794052291282,155.89794052291282,155.89794052291282,155.89794052291282,155.89794052291282,155.89794052291282,155.89794052291282,155.89794052291282,155.89794052291282,155.89794052291282,155.89794052291282,154.866733002167,154.866733002167,154.866733002167,154.866733002167,154.866733002167,154.866733002167,153.83552548142117,153.83552548142117,153.83552548142117,153.83552548142117,153.83552548142117,153.83552548142117,153.83552548142117,153.83552548142117,153.83552548142117,153.83552548142117,153.83552548142117,152.80431796067535,152.80431796067535,152.80431796067535,152.80431796067535,152.80431796067535,152.80431796067535,152.80431796067535,152.80431796067535,152.80431796067535,152.80431796067535,152.80431796067535,152.80431796067535,152.80431796067535,152.80431796067535,152.80431796067535,152.80431796067535,151.77311043992955,151.77311043992955,151.77311043992955,151.77311043992955,151.77311043992955,151.77311043992955,151.77311043992955,151.77311043992955,151.77311043992955,151.77311043992955,151.77311043992955,151.77311043992955,151.77311043992955,151.77311043992955,151.77311043992955,150.7419029191837,150.7419029191837,150.7419029191837,150.7419029191837,150.7419029191837,150.7419029191837,150.7419029191837,150.7419029191837,150.7419029191837,150.7419029191837,150.7419029191837,150.7419029191837,150.7419029191837,149.7106953984379,149.7106953984379,149.7106953984379,149.7106953984379,149.7106953984379,149.7106953984379,149.7106953984379,149.7106953984379,149.7106953984379,149.7106953984379,149.7106953984379,149.7106953984379,149.7106953984379,149.7106953984379,149.7106953984379,149.7106953984379,149.7106953984379,149.7106953984379,149.7106953984379,149.7106953984379,149.7106953984379,149.7106953984379,149.7106953984379,149.7106953984379,148.67948787769205,148.67948787769205,148.67948787769205,148.67948787769205,148.67948787769205,148.67948787769205,148.67948787769205,148.67948787769205,148.67948787769205,148.67948787769205,148.67948787769205,148.67948787769205,148.67948787769205,148.67948787769205,148.67948787769205,148.67948787769205,148.67948787769205,148.67948787769205,147.64828035694626,147.64828035694626,147.64828035694626,147.64828035694626,147.64828035694626,147.64828035694626,147.64828035694626,147.64828035694626,147.64828035694626,147.64828035694626,147.64828035694626,147.64828035694626,147.64828035694626,147.64828035694626,147.64828035694626,147.64828035694626,147.64828035694626,147.64828035694626,147.64828035694626,147.64828035694626,147.64828035694626,147.64828035694626,147.64828035694626,147.64828035694626,146.61707283620044,146.61707283620044,146.61707283620044,146.61707283620044,146.61707283620044,146.61707283620044,146.61707283620044,146.61707283620044,146.61707283620044,146.61707283620044,146.61707283620044,146.61707283620044,146.61707283620044,146.61707283620044,146.61707283620044,146.61707283620044,146.61707283620044,146.61707283620044,146.61707283620044,145.5858653154546,145.5858653154546,145.5858653154546,145.5858653154546,145.5858653154546,145.5858653154546,145.5858653154546,145.5858653154546,145.5858653154546,145.5858653154546,145.5858653154546,145.5858653154546,145.5858653154546,145.5858653154546,145.5858653154546,145.5858653154546,145.5858653154546,145.5858653154546,144.5546577947088,144.5546577947088,144.5546577947088,144.5546577947088,144.5546577947088,144.5546577947088,144.5546577947088,144.5546577947088,144.5546577947088,144.5546577947088,144.5546577947088,144.5546577947088,144.5546577947088,144.5546577947088,144.5546577947088,144.5546577947088,144.5546577947088,144.5546577947088,144.5546577947088,144.5546577947088,144.5546577947088,144.5546577947088,144.5546577947088,144.5546577947088,143.52345027396296,143.52345027396296,143.52345027396296,143.52345027396296,143.52345027396296,143.52345027396296,143.52345027396296,143.52345027396296,143.52345027396296,143.52345027396296,143.52345027396296,143.52345027396296,143.52345027396296,143.52345027396296,143.52345027396296,143.52345027396296,143.52345027396296,143.52345027396296,143.52345027396296,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,142.49224275321714,141.46103523247132,141.46103523247132,141.46103523247132,141.46103523247132,141.46103523247132,141.46103523247132,141.46103523247132,141.46103523247132,141.46103523247132,141.46103523247132,141.46103523247132,141.46103523247132,141.46103523247132,141.46103523247132,141.46103523247132,141.46103523247132,141.46103523247132,141.46103523247132,141.46103523247132,141.46103523247132,141.46103523247132,140.4298277117255,140.4298277117255,140.4298277117255,140.4298277117255,140.4298277117255,140.4298277117255,140.4298277117255,140.4298277117255,140.4298277117255,140.4298277117255,140.4298277117255,140.4298277117255,140.4298277117255,140.4298277117255,140.4298277117255,140.4298277117255,140.4298277117255,140.4298277117255,140.4298277117255,140.4298277117255,140.4298277117255,140.4298277117255,140.4298277117255,140.4298277117255,140.4298277117255,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,139.39862019097967,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,138.36741267023385,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,137.33620514948802,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,136.30499762874223,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,135.27379010799638,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,134.24258258725058,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,133.21137506650473,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,132.18016754575893,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,131.1489600250131,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,130.1177525042673,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,129.08654498352146,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,128.05533746277564,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,127.02412994202982,125.99292242128399,125.99292242128399,125.99292242128399,125.99292242128399,125.99292242128399,125.99292242128399,125.99292242128399,125.99292242128399,125.99292242128399,125.99292242128399,125.99292242128399,125.99292242128399,125.99292242128399,125.99292242128399,125.99292242128399,125.99292242128399,125.99292242128399,125.99292242128399,125.99292242128399,125.99292242128399,125.99292242128399,125.99292242128399,124.96171490053817,124.96171490053817,124.96171490053817,124.96171490053817,124.96171490053817,124.96171490053817,124.96171490053817,124.96171490053817,124.96171490053817,124.96171490053817,124.96171490053817,124.96171490053817,124.96171490053817,124.96171490053817,124.96171490053817,124.96171490053817,124.96171490053817,124.96171490053817,124.96171490053817,124.96171490053817,124.96171490053817,123.93050737979235,123.93050737979235,123.93050737979235,123.93050737979235,123.93050737979235,123.93050737979235,123.93050737979235,123.93050737979235,123.93050737979235,123.93050737979235,123.93050737979235,123.93050737979235,123.93050737979235,122.89929985904652,122.89929985904652,122.89929985904652,122.89929985904652,122.89929985904652,122.89929985904652,122.89929985904652,122.89929985904652,122.89929985904652,122.89929985904652,122.89929985904652,122.89929985904652,122.89929985904652,122.89929985904652,122.89929985904652,121.8680923383007,121.8680923383007,121.8680923383007,121.8680923383007,121.8680923383007,121.8680923383007,121.8680923383007,121.8680923383007,121.8680923383007,121.8680923383007,120.83688481755489,120.83688481755489,120.83688481755489,120.83688481755489,120.83688481755489,120.83688481755489,120.83688481755489,120.83688481755489,120.83688481755489,120.83688481755489,120.83688481755489,120.83688481755489,120.83688481755489,119.80567729680907,119.80567729680907,119.80567729680907,119.80567729680907,119.80567729680907,119.80567729680907,119.80567729680907,118.77446977606324,118.77446977606324,118.77446977606324,118.77446977606324,118.77446977606324,117.74326225531742,117.74326225531742,117.74326225531742,117.74326225531742,116.7120547345716,115.68084721382577,115.68084721382577,115.68084721382577,115.68084721382577,115.68084721382577,115.68084721382577,115.68084721382577,114.64963969307995,114.64963969307995,114.64963969307995,113.61843217233414,113.61843217233414,112.58722465158831,112.58722465158831],\"yaxis\":\"y\",\"type\":\"scatter\"}],                        {\"coloraxis\":{\"colorbar\":{\"title\":{\"text\":\"HeartDisease\"}},\"colorscale\":[[0.0,\"rgb(103,0,31)\"],[0.1,\"rgb(178,24,43)\"],[0.2,\"rgb(214,96,77)\"],[0.3,\"rgb(244,165,130)\"],[0.4,\"rgb(253,219,199)\"],[0.5,\"rgb(247,247,247)\"],[0.6,\"rgb(209,229,240)\"],[0.7,\"rgb(146,197,222)\"],[0.8,\"rgb(67,147,195)\"],[0.9,\"rgb(33,102,172)\"],[1.0,\"rgb(5,48,97)\"]]},\"hoverlabel\":{\"bgcolor\":\"#111\",\"font\":{\"family\":\"arial\",\"size\":16}},\"legend\":{\"tracegroupgap\":0},\"showlegend\":false,\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"white\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"#C8D4E3\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"white\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"radialaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"yaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"zaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"caxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2}}},\"title\":{\"font\":{\"family\":\"tahoma\",\"size\":30},\"text\":\"Max Heart Rate Via Age\"},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Age\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"MaxHR\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('ac2b45b2-f8b1-4cff-b680-19598eaa4311');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = px.scatter(\n",
    "    data_frame=df,\n",
    "    x = \"Age\",\n",
    "    y = \"MaxHR\",\n",
    "    trendline=\"ols\",\n",
    "    color = \"HeartDisease\",\n",
    "    template=\"plotly_white\",\n",
    "    color_continuous_scale=\"RdBu\",\n",
    "    title=\"Max Heart Rate Via Age\"\n",
    ")\n",
    "\n",
    "custome_layout(title_size=30, showlegend=False)\n",
    "\n",
    "iplot(fig)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c3a79415",
   "metadata": {
    "papermill": {
     "duration": 0.051421,
     "end_time": "2024-03-10T10:04:56.992202",
     "exception": false,
     "start_time": "2024-03-10T10:04:56.940781",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Time To Build Our Model 👷‍♂️"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8b65c6ad",
   "metadata": {
    "papermill": {
     "duration": 0.050387,
     "end_time": "2024-03-10T10:04:57.093799",
     "exception": false,
     "start_time": "2024-03-10T10:04:57.043412",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<p style = \"color: #00FFAB;\n",
    "            font: bold 22px tahoma;\n",
    "            background-color: #111;\n",
    "            padding: 18px;\n",
    "            border: 3px solid lightgreen;\n",
    "            border-radius: 8px\"> \n",
    "    1] Categorical Data Encoding \n",
    "</p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "fcf33adc",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:57.199494Z",
     "iopub.status.busy": "2024-03-10T10:04:57.198121Z",
     "iopub.status.idle": "2024-03-10T10:04:57.207462Z",
     "shell.execute_reply": "2024-03-10T10:04:57.206395Z"
    },
    "papermill": {
     "duration": 0.06339,
     "end_time": "2024-03-10T10:04:57.209822",
     "exception": false,
     "start_time": "2024-03-10T10:04:57.146432",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], dtype='object')"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.select_dtypes(include=\"object\").columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "f9c99027",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:57.312836Z",
     "iopub.status.busy": "2024-03-10T10:04:57.312081Z",
     "iopub.status.idle": "2024-03-10T10:04:57.341297Z",
     "shell.execute_reply": "2024-03-10T10:04:57.339930Z"
    },
    "papermill": {
     "duration": 0.085423,
     "end_time": "2024-03-10T10:04:57.344447",
     "exception": false,
     "start_time": "2024-03-10T10:04:57.259024",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>RestingBP</th>\n",
       "      <th>Cholesterol</th>\n",
       "      <th>FastingBS</th>\n",
       "      <th>MaxHR</th>\n",
       "      <th>Oldpeak</th>\n",
       "      <th>HeartDisease</th>\n",
       "      <th>Sex_M</th>\n",
       "      <th>ChestPainType_ATA</th>\n",
       "      <th>ChestPainType_NAP</th>\n",
       "      <th>ChestPainType_TA</th>\n",
       "      <th>RestingECG_Normal</th>\n",
       "      <th>RestingECG_ST</th>\n",
       "      <th>ExerciseAngina_Y</th>\n",
       "      <th>ST_Slope_Flat</th>\n",
       "      <th>ST_Slope_Up</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>40</td>\n",
       "      <td>140</td>\n",
       "      <td>289</td>\n",
       "      <td>0</td>\n",
       "      <td>172</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>49</td>\n",
       "      <td>160</td>\n",
       "      <td>180</td>\n",
       "      <td>0</td>\n",
       "      <td>156</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>37</td>\n",
       "      <td>130</td>\n",
       "      <td>283</td>\n",
       "      <td>0</td>\n",
       "      <td>98</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>48</td>\n",
       "      <td>138</td>\n",
       "      <td>214</td>\n",
       "      <td>0</td>\n",
       "      <td>108</td>\n",
       "      <td>1.5</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>54</td>\n",
       "      <td>150</td>\n",
       "      <td>195</td>\n",
       "      <td>0</td>\n",
       "      <td>122</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>913</th>\n",
       "      <td>45</td>\n",
       "      <td>110</td>\n",
       "      <td>264</td>\n",
       "      <td>0</td>\n",
       "      <td>132</td>\n",
       "      <td>1.2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>914</th>\n",
       "      <td>68</td>\n",
       "      <td>144</td>\n",
       "      <td>193</td>\n",
       "      <td>1</td>\n",
       "      <td>141</td>\n",
       "      <td>3.4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>915</th>\n",
       "      <td>57</td>\n",
       "      <td>130</td>\n",
       "      <td>131</td>\n",
       "      <td>0</td>\n",
       "      <td>115</td>\n",
       "      <td>1.2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>916</th>\n",
       "      <td>57</td>\n",
       "      <td>130</td>\n",
       "      <td>236</td>\n",
       "      <td>0</td>\n",
       "      <td>174</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>917</th>\n",
       "      <td>38</td>\n",
       "      <td>138</td>\n",
       "      <td>175</td>\n",
       "      <td>0</td>\n",
       "      <td>173</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>918 rows × 16 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Age  RestingBP  Cholesterol  FastingBS  MaxHR  Oldpeak  HeartDisease  \\\n",
       "0     40        140          289          0    172      0.0             0   \n",
       "1     49        160          180          0    156      1.0             1   \n",
       "2     37        130          283          0     98      0.0             0   \n",
       "3     48        138          214          0    108      1.5             1   \n",
       "4     54        150          195          0    122      0.0             0   \n",
       "..   ...        ...          ...        ...    ...      ...           ...   \n",
       "913   45        110          264          0    132      1.2             1   \n",
       "914   68        144          193          1    141      3.4             1   \n",
       "915   57        130          131          0    115      1.2             1   \n",
       "916   57        130          236          0    174      0.0             1   \n",
       "917   38        138          175          0    173      0.0             0   \n",
       "\n",
       "     Sex_M  ChestPainType_ATA  ChestPainType_NAP  ChestPainType_TA  \\\n",
       "0        1                  1                  0                 0   \n",
       "1        0                  0                  1                 0   \n",
       "2        1                  1                  0                 0   \n",
       "3        0                  0                  0                 0   \n",
       "4        1                  0                  1                 0   \n",
       "..     ...                ...                ...               ...   \n",
       "913      1                  0                  0                 1   \n",
       "914      1                  0                  0                 0   \n",
       "915      1                  0                  0                 0   \n",
       "916      0                  1                  0                 0   \n",
       "917      1                  0                  1                 0   \n",
       "\n",
       "     RestingECG_Normal  RestingECG_ST  ExerciseAngina_Y  ST_Slope_Flat  \\\n",
       "0                    1              0                 0              0   \n",
       "1                    1              0                 0              1   \n",
       "2                    0              1                 0              0   \n",
       "3                    1              0                 1              1   \n",
       "4                    1              0                 0              0   \n",
       "..                 ...            ...               ...            ...   \n",
       "913                  1              0                 0              1   \n",
       "914                  1              0                 0              1   \n",
       "915                  1              0                 1              1   \n",
       "916                  0              0                 0              1   \n",
       "917                  1              0                 0              0   \n",
       "\n",
       "     ST_Slope_Up  \n",
       "0              1  \n",
       "1              0  \n",
       "2              1  \n",
       "3              0  \n",
       "4              1  \n",
       "..           ...  \n",
       "913            0  \n",
       "914            0  \n",
       "915            0  \n",
       "916            0  \n",
       "917            1  \n",
       "\n",
       "[918 rows x 16 columns]"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_encodded = pd.get_dummies(data=df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], \n",
    "                             drop_first=\"True\")*1\n",
    "\n",
    "df_encodded"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6f0f5eb2",
   "metadata": {
    "papermill": {
     "duration": 0.051724,
     "end_time": "2024-03-10T10:04:57.451669",
     "exception": false,
     "start_time": "2024-03-10T10:04:57.399945",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<p style = \"color: #00FFAB;\n",
    "            font: bold 22px tahoma;\n",
    "            background-color: #111;\n",
    "            padding: 18px;\n",
    "            border: 3px solid lightgreen;\n",
    "            border-radius: 8px\"> \n",
    "    2] Splitting Data Into Train & Test\n",
    "</p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "a244d1fb",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:57.564243Z",
     "iopub.status.busy": "2024-03-10T10:04:57.563546Z",
     "iopub.status.idle": "2024-03-10T10:04:57.570243Z",
     "shell.execute_reply": "2024-03-10T10:04:57.569056Z"
    },
    "papermill": {
     "duration": 0.066776,
     "end_time": "2024-03-10T10:04:57.572966",
     "exception": false,
     "start_time": "2024-03-10T10:04:57.506190",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "X = df_encodded.drop(columns=[\"HeartDisease\"])\n",
    "y = df_encodded[\"HeartDisease\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "37d4381e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:57.681610Z",
     "iopub.status.busy": "2024-03-10T10:04:57.681191Z",
     "iopub.status.idle": "2024-03-10T10:04:57.689520Z",
     "shell.execute_reply": "2024-03-10T10:04:57.688562Z"
    },
    "papermill": {
     "duration": 0.065184,
     "end_time": "2024-03-10T10:04:57.691814",
     "exception": false,
     "start_time": "2024-03-10T10:04:57.626630",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6bb2b30a",
   "metadata": {
    "papermill": {
     "duration": 0.052327,
     "end_time": "2024-03-10T10:04:57.796548",
     "exception": false,
     "start_time": "2024-03-10T10:04:57.744221",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<p style = \"color: #00FFAB;\n",
    "            font: bold 22px tahoma;\n",
    "            background-color: #111;\n",
    "            padding: 18px;\n",
    "            border: 3px solid lightgreen;\n",
    "            border-radius: 8px\"> \n",
    "    3] Hyperparameter Tuning Using RandomizedSearchCV👩‍💻🚀\n",
    "</p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "4c19681a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:57.899884Z",
     "iopub.status.busy": "2024-03-10T10:04:57.899030Z",
     "iopub.status.idle": "2024-03-10T10:04:57.905894Z",
     "shell.execute_reply": "2024-03-10T10:04:57.905056Z"
    },
    "papermill": {
     "duration": 0.060425,
     "end_time": "2024-03-10T10:04:57.908236",
     "exception": false,
     "start_time": "2024-03-10T10:04:57.847811",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Create The Model\n",
    "model = XGBClassifier()\n",
    "\n",
    "# Used Hyperparameter\n",
    "param_grid = {\n",
    "    'learning_rate': [0.01, 0.1, 0.2, 0.3],\n",
    "    'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],\n",
    "    'gamma': [0, 0.25, 0.4, 0.5, 1.0],\n",
    "    'min_child_weight': [1, 3, 5, 7],\n",
    "    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],\n",
    "    'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "fe2716a7",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:58.011951Z",
     "iopub.status.busy": "2024-03-10T10:04:58.011069Z",
     "iopub.status.idle": "2024-03-10T10:04:58.016563Z",
     "shell.execute_reply": "2024-03-10T10:04:58.015326Z"
    },
    "papermill": {
     "duration": 0.060634,
     "end_time": "2024-03-10T10:04:58.018988",
     "exception": false,
     "start_time": "2024-03-10T10:04:57.958354",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Verbose = 0 To Hide Results of all Tuning\n",
    "random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=35, \n",
    "                                   scoring=\"roc_auc\", cv=5, verbose=0, random_state=99)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "8d586696",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:04:58.119321Z",
     "iopub.status.busy": "2024-03-10T10:04:58.118770Z",
     "iopub.status.idle": "2024-03-10T10:05:14.824019Z",
     "shell.execute_reply": "2024-03-10T10:05:14.823024Z"
    },
    "papermill": {
     "duration": 16.759503,
     "end_time": "2024-03-10T10:05:14.827699",
     "exception": false,
     "start_time": "2024-03-10T10:04:58.068196",
     "status": "completed"
    },
    "scrolled": true,
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomizedSearchCV(cv=5,\n",
       "                   estimator=XGBClassifier(base_score=None, booster=None,\n",
       "                                           callbacks=None,\n",
       "                                           colsample_bylevel=None,\n",
       "                                           colsample_bynode=None,\n",
       "                                           colsample_bytree=None, device=None,\n",
       "                                           early_stopping_rounds=None,\n",
       "                                           enable_categorical=False,\n",
       "                                           eval_metric=None, feature_types=None,\n",
       "                                           gamma=None, grow_policy=None,\n",
       "                                           importance_type=None,\n",
       "                                           interaction_constraints=None,\n",
       "                                           learning_rate...\n",
       "                                           n_estimators=None, n_jobs=None,\n",
       "                                           num_parallel_tree=None,\n",
       "                                           random_state=None, ...),\n",
       "                   n_iter=35,\n",
       "                   param_distributions={&#x27;colsample_bytree&#x27;: [0.4, 0.5, 0.6, 0.7,\n",
       "                                                             0.8, 0.9, 1.0],\n",
       "                                        &#x27;gamma&#x27;: [0, 0.25, 0.4, 0.5, 1.0],\n",
       "                                        &#x27;learning_rate&#x27;: [0.01, 0.1, 0.2, 0.3],\n",
       "                                        &#x27;max_depth&#x27;: [3, 4, 5, 6, 8, 10, 12,\n",
       "                                                      15],\n",
       "                                        &#x27;min_child_weight&#x27;: [1, 3, 5, 7],\n",
       "                                        &#x27;subsample&#x27;: [0.5, 0.6, 0.7, 0.8, 0.9,\n",
       "                                                      1.0]},\n",
       "                   random_state=99, scoring=&#x27;roc_auc&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" ><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomizedSearchCV</label><div class=\"sk-toggleable__content\"><pre>RandomizedSearchCV(cv=5,\n",
       "                   estimator=XGBClassifier(base_score=None, booster=None,\n",
       "                                           callbacks=None,\n",
       "                                           colsample_bylevel=None,\n",
       "                                           colsample_bynode=None,\n",
       "                                           colsample_bytree=None, device=None,\n",
       "                                           early_stopping_rounds=None,\n",
       "                                           enable_categorical=False,\n",
       "                                           eval_metric=None, feature_types=None,\n",
       "                                           gamma=None, grow_policy=None,\n",
       "                                           importance_type=None,\n",
       "                                           interaction_constraints=None,\n",
       "                                           learning_rate...\n",
       "                                           n_estimators=None, n_jobs=None,\n",
       "                                           num_parallel_tree=None,\n",
       "                                           random_state=None, ...),\n",
       "                   n_iter=35,\n",
       "                   param_distributions={&#x27;colsample_bytree&#x27;: [0.4, 0.5, 0.6, 0.7,\n",
       "                                                             0.8, 0.9, 1.0],\n",
       "                                        &#x27;gamma&#x27;: [0, 0.25, 0.4, 0.5, 1.0],\n",
       "                                        &#x27;learning_rate&#x27;: [0.01, 0.1, 0.2, 0.3],\n",
       "                                        &#x27;max_depth&#x27;: [3, 4, 5, 6, 8, 10, 12,\n",
       "                                                      15],\n",
       "                                        &#x27;min_child_weight&#x27;: [1, 3, 5, 7],\n",
       "                                        &#x27;subsample&#x27;: [0.5, 0.6, 0.7, 0.8, 0.9,\n",
       "                                                      1.0]},\n",
       "                   random_state=99, scoring=&#x27;roc_auc&#x27;)</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" ><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">estimator: XGBClassifier</label><div class=\"sk-toggleable__content\"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,\n",
       "              colsample_bylevel=None, colsample_bynode=None,\n",
       "              colsample_bytree=None, device=None, early_stopping_rounds=None,\n",
       "              enable_categorical=False, eval_metric=None, feature_types=None,\n",
       "              gamma=None, grow_policy=None, importance_type=None,\n",
       "              interaction_constraints=None, learning_rate=None, max_bin=None,\n",
       "              max_cat_threshold=None, max_cat_to_onehot=None,\n",
       "              max_delta_step=None, max_depth=None, max_leaves=None,\n",
       "              min_child_weight=None, missing=nan, monotone_constraints=None,\n",
       "              multi_strategy=None, n_estimators=None, n_jobs=None,\n",
       "              num_parallel_tree=None, random_state=None, ...)</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-3\" type=\"checkbox\" ><label for=\"sk-estimator-id-3\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">XGBClassifier</label><div class=\"sk-toggleable__content\"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,\n",
       "              colsample_bylevel=None, colsample_bynode=None,\n",
       "              colsample_bytree=None, device=None, early_stopping_rounds=None,\n",
       "              enable_categorical=False, eval_metric=None, feature_types=None,\n",
       "              gamma=None, grow_policy=None, importance_type=None,\n",
       "              interaction_constraints=None, learning_rate=None, max_bin=None,\n",
       "              max_cat_threshold=None, max_cat_to_onehot=None,\n",
       "              max_delta_step=None, max_depth=None, max_leaves=None,\n",
       "              min_child_weight=None, missing=nan, monotone_constraints=None,\n",
       "              multi_strategy=None, n_estimators=None, n_jobs=None,\n",
       "              num_parallel_tree=None, random_state=None, ...)</pre></div></div></div></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "RandomizedSearchCV(cv=5,\n",
       "                   estimator=XGBClassifier(base_score=None, booster=None,\n",
       "                                           callbacks=None,\n",
       "                                           colsample_bylevel=None,\n",
       "                                           colsample_bynode=None,\n",
       "                                           colsample_bytree=None, device=None,\n",
       "                                           early_stopping_rounds=None,\n",
       "                                           enable_categorical=False,\n",
       "                                           eval_metric=None, feature_types=None,\n",
       "                                           gamma=None, grow_policy=None,\n",
       "                                           importance_type=None,\n",
       "                                           interaction_constraints=None,\n",
       "                                           learning_rate...\n",
       "                                           n_estimators=None, n_jobs=None,\n",
       "                                           num_parallel_tree=None,\n",
       "                                           random_state=None, ...),\n",
       "                   n_iter=35,\n",
       "                   param_distributions={'colsample_bytree': [0.4, 0.5, 0.6, 0.7,\n",
       "                                                             0.8, 0.9, 1.0],\n",
       "                                        'gamma': [0, 0.25, 0.4, 0.5, 1.0],\n",
       "                                        'learning_rate': [0.01, 0.1, 0.2, 0.3],\n",
       "                                        'max_depth': [3, 4, 5, 6, 8, 10, 12,\n",
       "                                                      15],\n",
       "                                        'min_child_weight': [1, 3, 5, 7],\n",
       "                                        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9,\n",
       "                                                      1.0]},\n",
       "                   random_state=99, scoring='roc_auc')"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "random_search.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "8589a239",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:05:14.936023Z",
     "iopub.status.busy": "2024-03-10T10:05:14.935172Z",
     "iopub.status.idle": "2024-03-10T10:05:14.941761Z",
     "shell.execute_reply": "2024-03-10T10:05:14.940398Z"
    },
    "papermill": {
     "duration": 0.063522,
     "end_time": "2024-03-10T10:05:14.944204",
     "exception": false,
     "start_time": "2024-03-10T10:05:14.880682",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "subsample=0.7,\n",
      "min_child_weight=3,\n",
      "max_depth=3,\n",
      "learning_rate=0.1,\n",
      "gamma=0,\n",
      "colsample_bytree=0.5,\n"
     ]
    }
   ],
   "source": [
    "p = random_search.best_params_\n",
    "for k, v in p.items():\n",
    "    print(f\"{k}={v},\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "578eb3a4",
   "metadata": {
    "papermill": {
     "duration": 0.053427,
     "end_time": "2024-03-10T10:05:15.050681",
     "exception": false,
     "start_time": "2024-03-10T10:05:14.997254",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<p style = \"color: #00FFAB;\n",
    "            font: bold 22px tahoma;\n",
    "            background-color: #111;\n",
    "            padding: 18px;\n",
    "            border: 3px solid lightgreen;\n",
    "            border-radius: 8px\"> \n",
    "    4] Build XGBoost Model With Best Parametrs 👩‍💻🚀\n",
    "</p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "9e6e5dc8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:05:15.159302Z",
     "iopub.status.busy": "2024-03-10T10:05:15.157791Z",
     "iopub.status.idle": "2024-03-10T10:05:15.164701Z",
     "shell.execute_reply": "2024-03-10T10:05:15.163443Z"
    },
    "papermill": {
     "duration": 0.064288,
     "end_time": "2024-03-10T10:05:15.167644",
     "exception": false,
     "start_time": "2024-03-10T10:05:15.103356",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "model = XGBClassifier(\n",
    "objective=\"binary:logistic\",\n",
    "subsample=0.7,\n",
    "min_child_weight=3,\n",
    "max_depth=3,\n",
    "learning_rate=0.1,\n",
    "gamma=0,\n",
    "colsample_bytree=0.5,\n",
    ")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f5143b97",
   "metadata": {
    "papermill": {
     "duration": 0.054075,
     "end_time": "2024-03-10T10:05:15.274409",
     "exception": false,
     "start_time": "2024-03-10T10:05:15.220334",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<p style = \"color: #00FFAB;\n",
    "            font: bold 22px tahoma;\n",
    "            background-color: #111;\n",
    "            padding: 18px;\n",
    "            border: 3px solid lightgreen;\n",
    "            border-radius: 8px\"> \n",
    "    5] Cross Validation Score 👩‍💻🚀\n",
    "</p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "cec309e0",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:05:15.387795Z",
     "iopub.status.busy": "2024-03-10T10:05:15.386734Z",
     "iopub.status.idle": "2024-03-10T10:05:16.169326Z",
     "shell.execute_reply": "2024-03-10T10:05:16.168205Z"
    },
    "papermill": {
     "duration": 0.843641,
     "end_time": "2024-03-10T10:05:16.172308",
     "exception": false,
     "start_time": "2024-03-10T10:05:15.328667",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CROSS VALIDATION SCORE: 88.23%\n"
     ]
    }
   ],
   "source": [
    "# Cross Validation Score\n",
    "kf = KFold(n_splits=10, shuffle=True, random_state=42)\n",
    "scores = cross_val_score(model, X, y, cv = kf)\n",
    "\n",
    "print(f\"CROSS VALIDATION SCORE: {np.mean(scores)*100:0.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "45359c4e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:05:16.285532Z",
     "iopub.status.busy": "2024-03-10T10:05:16.285070Z",
     "iopub.status.idle": "2024-03-10T10:05:16.367509Z",
     "shell.execute_reply": "2024-03-10T10:05:16.366269Z"
    },
    "papermill": {
     "duration": 0.143575,
     "end_time": "2024-03-10T10:05:16.370258",
     "exception": false,
     "start_time": "2024-03-10T10:05:16.226683",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "TRAIN SCORE 91.14%\n"
     ]
    }
   ],
   "source": [
    "model.fit(X_train, y_train)\n",
    "train_score = model.score(X_train, y_train)*100\n",
    "print(f\"TRAIN SCORE {train_score:0.2f}%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "21772bd8",
   "metadata": {
    "papermill": {
     "duration": 0.049986,
     "end_time": "2024-03-10T10:05:16.471663",
     "exception": false,
     "start_time": "2024-03-10T10:05:16.421677",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<p style = \"color: #00FFAB;\n",
    "            font: bold 22px tahoma;\n",
    "            background-color: #111;\n",
    "            padding: 18px;\n",
    "            border: 3px solid lightgreen;\n",
    "            border-radius: 8px\"> \n",
    "    6] Test Our Model 👩‍💻🚀\n",
    "</p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "817e112f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:05:16.574384Z",
     "iopub.status.busy": "2024-03-10T10:05:16.573627Z",
     "iopub.status.idle": "2024-03-10T10:05:16.594308Z",
     "shell.execute_reply": "2024-03-10T10:05:16.593096Z"
    },
    "papermill": {
     "duration": 0.075945,
     "end_time": "2024-03-10T10:05:16.597583",
     "exception": false,
     "start_time": "2024-03-10T10:05:16.521638",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "TEST SCORE 92.93%\n"
     ]
    }
   ],
   "source": [
    "predictions = model.predict(X_test)\n",
    "test_score = accuracy_score(y_test, predictions)*100\n",
    "print(f\"TEST SCORE {test_score:0.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "8409a9f7",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:05:16.712062Z",
     "iopub.status.busy": "2024-03-10T10:05:16.710710Z",
     "iopub.status.idle": "2024-03-10T10:05:16.726194Z",
     "shell.execute_reply": "2024-03-10T10:05:16.725198Z"
    },
    "papermill": {
     "duration": 0.073422,
     "end_time": "2024-03-10T10:05:16.728714",
     "exception": false,
     "start_time": "2024-03-10T10:05:16.655292",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.92      0.93      0.92        86\n",
      "           1       0.94      0.93      0.93        98\n",
      "\n",
      "    accuracy                           0.93       184\n",
      "   macro avg       0.93      0.93      0.93       184\n",
      "weighted avg       0.93      0.93      0.93       184\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test, predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "b46b3760",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:05:16.833332Z",
     "iopub.status.busy": "2024-03-10T10:05:16.832465Z",
     "iopub.status.idle": "2024-03-10T10:05:16.926385Z",
     "shell.execute_reply": "2024-03-10T10:05:16.925196Z"
    },
    "papermill": {
     "duration": 0.149201,
     "end_time": "2024-03-10T10:05:16.928999",
     "exception": false,
     "start_time": "2024-03-10T10:05:16.779798",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>                            <div id=\"3511eb01-a2c7-447d-b119-abaaf0ae668f\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"3511eb01-a2c7-447d-b119-abaaf0ae668f\")) {                    Plotly.newPlot(                        \"3511eb01-a2c7-447d-b119-abaaf0ae668f\",                        [{\"coloraxis\":\"coloraxis\",\"hovertemplate\":\"x: %{x}\\u003cbr\\u003ey: %{y}\\u003cbr\\u003ecolor: %{z}\\u003cextra\\u003e\\u003c\\u002fextra\\u003e\",\"name\":\"0\",\"texttemplate\":\"%{z}\",\"x\":[\"No Heart Disease\",\"Heart Disease\"],\"xaxis\":\"x\",\"y\":[\"No Heart Disease\",\"Heart Disease\"],\"yaxis\":\"y\",\"z\":[[80,6],[7,91]],\"type\":\"heatmap\"}],                        {\"coloraxis\":{\"colorscale\":[[0.0,\"rgb(103,0,31)\"],[0.1,\"rgb(178,24,43)\"],[0.2,\"rgb(214,96,77)\"],[0.3,\"rgb(244,165,130)\"],[0.4,\"rgb(253,219,199)\"],[0.5,\"rgb(247,247,247)\"],[0.6,\"rgb(209,229,240)\"],[0.7,\"rgb(146,197,222)\"],[0.8,\"rgb(67,147,195)\"],[0.9,\"rgb(33,102,172)\"],[1.0,\"rgb(5,48,97)\"]]},\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"white\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2}}},\"title\":{\"text\":\"Confusion Matrix\"},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0]},\"yaxis\":{\"anchor\":\"x\",\"autorange\":\"reversed\",\"domain\":[0.0,1.0]}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('3511eb01-a2c7-447d-b119-abaaf0ae668f');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "cm = confusion_matrix(y_test, predictions)\n",
    "ticks = df[\"HeartDisease\"].map({0:\"No Heart Disease\", 1:\"Heart Disease\"}).unique()\n",
    "\n",
    "fig = px.imshow(cm, aspect=True, text_auto=True, x=ticks, y=ticks, \n",
    "          color_continuous_scale=\"RdBu\", title=\"Confusion Matrix\")\n",
    "iplot(fig)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0a0177b1",
   "metadata": {
    "papermill": {
     "duration": 0.052138,
     "end_time": "2024-03-10T10:05:17.032795",
     "exception": false,
     "start_time": "2024-03-10T10:05:16.980657",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<p style = \"color: #247881;\n",
    "            font: bold 20px tahoma;\n",
    "            background-color: #fff;\n",
    "            padding: 18px;\n",
    "            border: 6px solid #247881;\n",
    "            border-radius: 8px\"> \n",
    "    🚀 Cross Validation Score: 88.23%\n",
    "    <br>\n",
    "    <br>\n",
    "    🚀 Training Score: 91.14%\n",
    "    <br>\n",
    "    <br>\n",
    "    🚀 Test Score: 92.93%\n",
    "    \n",
    "</p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "03b3d786",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-10T10:05:17.138505Z",
     "iopub.status.busy": "2024-03-10T10:05:17.137994Z",
     "iopub.status.idle": "2024-03-10T10:05:17.142866Z",
     "shell.execute_reply": "2024-03-10T10:05:17.141933Z"
    },
    "papermill": {
     "duration": 0.060304,
     "end_time": "2024-03-10T10:05:17.145070",
     "exception": false,
     "start_time": "2024-03-10T10:05:17.084766",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# pd.to_pickle(model, \"xgboost_hear_disease_detection_v1.pkl\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "37215f61",
   "metadata": {
    "papermill": {
     "duration": 0.051241,
     "end_time": "2024-03-10T10:05:17.247937",
     "exception": false,
     "start_time": "2024-03-10T10:05:17.196696",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# ♠ Streamlit Web APP 🤩🚀"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "91838fb9",
   "metadata": {
    "papermill": {
     "duration": 0.051106,
     "end_time": "2024-03-10T10:05:17.350851",
     "exception": false,
     "start_time": "2024-03-10T10:05:17.299745",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Web App Link: <a href = \"https://heart-failure-prediction-8lygyzhmwy3uctejsac3wj.streamlit.app/\" style = \"color: red\">Click Here To Visit Streamlit Web App</a>🥰📊"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "45a3c291",
   "metadata": {
    "papermill": {
     "duration": 0.051271,
     "end_time": "2024-03-10T10:05:17.454577",
     "exception": false,
     "start_time": "2024-03-10T10:05:17.403306",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## GitHub Repo: <a href = \"https://github.com/modyehab810/Heart-Failure-Prediction\">Click Here To Go To GitHub Repo..</a>"
   ]
  },
  {
   "attachments": {
    "image.png": {
     "image/png": "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"
    }
   },
   "cell_type": "markdown",
   "id": "0fd525aa",
   "metadata": {
    "papermill": {
     "duration": 0.050826,
     "end_time": "2024-03-10T10:05:17.556107",
     "exception": false,
     "start_time": "2024-03-10T10:05:17.505281",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "![image.png](attachment:image.png)"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "datasetId": 1582403,
     "sourceId": 2603715,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 30664,
   "isGpuEnabled": false,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.13"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 43.516913,
   "end_time": "2024-03-10T10:05:18.632952",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2024-03-10T10:04:35.116039",
   "version": "2.5.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
