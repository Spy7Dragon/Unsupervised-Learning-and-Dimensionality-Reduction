from kmeans_clustering import *
from expectation_maximization import *
from pca import *
from ica import *
from rca import *
from any_feature_selection import *
from neural_network import *
import pandas as pd
import numpy as np

columns = [
    "annual_inc",
    "collections_12_mths_ex_med",
    "delinq_amnt",
    "delinq_2yrs",
    "dti",
    "fico_range_high",
    "fico_range_low",
    "home_ownership",
    "inq_last_6mths",
    "installment",
    "int_rate",
    "verification_status",
    "loan_amnt",
    "mths_since_last_major_derog",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "purpose",
    "revol_bal",
    "revol_util",
    "sub_grade",
    "term",
    "total_acc",
    "loan_status"
]

feature_columns = ['SEX', 'SCHL', 'OCCP', 'PINCP']


def get_lending_club_data():
    data = {}
    train = pd.read_csv("data/lendingclub_train.csv", usecols=columns)
    test = pd.read_csv("data/lendingclub_test.csv", usecols=columns)

    training_features = train[columns[:-1]]
    training_classes = train[columns[-1:]].astype(np.bool)
    test_features = test[columns[:-1]]
    test_classes = test[columns[-1:]].astype(np.bool)

    data['training_features'] = training_features
    data['training_classes'] = training_classes
    data['test_features'] = test_features
    data['test_classes'] = test_classes
    data['directory'] = 'lendingclub_observations'
    data['table_directory'] = 'lendingclub_tables'
    return data


def get_divorce_data():
    data = {}
    training_features = pd.read_csv('data/divorce_train_features.csv', usecols=feature_columns)
    training_classes = pd.read_csv('data/divorce_train_classes.csv', index_col=0)
    test_features = pd.read_csv('data/divorce_test_features.csv', usecols=feature_columns)
    test_classes = pd.read_csv('data/divorce_test_classes.csv', index_col=0)

    data['training_features'] = training_features
    data['training_classes'] = training_classes
    data['test_features'] = test_features
    data['test_classes'] = test_classes
    data['directory'] = 'divorce_observations'
    data['table_directory'] = 'divorce_tables'
    return data


def perform_on_data(data):
    data['dimensionality_reduction'] = None
    data['clusterer'] = None
    # 1. Run the clustering algorithms on the datasets and describe what you see.
    perform_kmeans_clustering(data)
    perform_expectation_maximization(data)
    # 2. Apply the dimensionality reduction algorithms to the two datasets
    # and describe what you see.
    pca_training_features, pca_test_features = perform_pca(data)
    ica_training_features, ica_test_features = perform_ica(data)
    rca_training_features, rca_test_features = perform_rca(data)
    any_training_features, any_test_features = perform_any_feature_selection(data)
    # 3. Reproduce your clustering experiments,
    # but on the data after you've run dimensionality reduction on it.
    data['dimensionality_reduction'] = 'PCA'
    data['training_features'] = pca_training_features
    data['test_features'] = pca_test_features
    pca_kmeans_training_features, pca_kmeans_test_features = perform_kmeans_clustering(data)
    pca_em_training_features, pca_em_test_features = perform_expectation_maximization(data)
    data['dimensionality_reduction'] = 'ICA'
    data['training_features'] = ica_training_features
    data['test_features'] = ica_test_features
    ica_kmeans_training_features, ica_kmeans_test_features = perform_kmeans_clustering(data)
    ica_em_training_features, ica_em_test_features = perform_expectation_maximization(data)
    data['dimensionality_reduction'] = 'RCA'
    data['training_features'] = rca_training_features
    data['test_features'] = rca_test_features
    rca_kmeans_training_features, rca_kmeans_test_features = perform_kmeans_clustering(data)
    rca_em_training_features, rca_em_test_features = perform_expectation_maximization(data)
    data['dimensionality_reduction'] = 'ANY'
    data['training_features'] = any_training_features
    data['test_features'] = any_test_features
    any_kmeans_training_features, any_kmeans_test_features = perform_kmeans_clustering(data)
    any_em_training_features, any_em_test_features = perform_expectation_maximization(data)
    # 4. Apply the dimensionality reduction algorithms
    # to one of your datasets from assignment #1
    # and rerun your neural network learner on the newly projected data.
    data['dimensionality_reduction'] = 'PCA'
    data['training_features'] = pca_training_features
    data['test_features'] = pca_test_features
    perform_neural_network(data)
    data['dimensionality_reduction'] = 'ICA'
    data['training_features'] = ica_training_features
    data['test_features'] = ica_test_features
    perform_neural_network(data)
    data['dimensionality_reduction'] = 'RCA'
    data['training_features'] = rca_training_features
    data['test_features'] = rca_test_features
    perform_neural_network(data)
    data['dimensionality_reduction'] = 'ANY'
    data['training_features'] = any_training_features
    data['test_features'] = any_test_features
    perform_neural_network(data)
    # 5. Apply the clustering algorithms to the same dataset
    # to which you just applied the dimensionality reduction algorithms
    # (you've probably already done this),
    # treating the clusters as if they were new features.
    # In other words, treat the clustering algorithms
    # as if they were dimensionality reduction algorithms.
    # Again, rerun your neural network learner on the newly projected data.
    data['dimensionality_reduction'] = 'PCA'
    data['clusterer'] = 'KMEANS'
    data['training_features'] = pca_kmeans_training_features
    data['test_features'] = pca_kmeans_test_features
    perform_neural_network(data)
    data['clusterer'] = 'EM'
    data['training_features'] = pca_em_training_features
    data['test_features'] = pca_em_test_features
    perform_neural_network(data)
    data['dimensionality_reduction'] = 'ICA'
    data['clusterer'] = 'KMEANS'
    data['training_features'] = ica_kmeans_training_features
    data['test_features'] = ica_kmeans_test_features
    perform_neural_network(data)
    data['clusterer'] = 'EM'
    data['training_features'] = ica_em_training_features
    data['test_features'] = ica_em_test_features
    perform_neural_network(data)
    data['dimensionality_reduction'] = 'RCA'
    data['clusterer'] = 'KMEANS'
    data['training_features'] = rca_kmeans_training_features
    data['test_features'] = rca_kmeans_test_features
    perform_neural_network(data)
    data['clusterer'] = 'EM'
    data['training_features'] = rca_em_training_features
    data['test_features'] = rca_em_test_features
    perform_neural_network(data)
    data['dimensionality_reduction'] = 'ANY'
    data['clusterer'] = 'KMEANS'
    data['training_features'] = any_kmeans_training_features
    data['test_features'] = any_kmeans_test_features
    perform_neural_network(data)
    data['clusterer'] = 'EM'
    data['training_features'] = any_em_training_features
    data['test_features'] = any_em_test_features
    perform_neural_network(data)


if __name__ == '__main__':
    data = get_lending_club_data()
    perform_on_data(data)
    data = get_divorce_data()
    perform_on_data(data)

# MAR Character 1
# Marital status
# 1 .Married
# 2 .Widowed
# 3 .Divorced
# 4 .Separated
# Remove 5 .Never married or under 15 years old

# MARHT Character 1
# Number of times married
# b .N/A (age less than 15 years; never married)
# 1 .One time
# Remove 2 .Two times
# Remove 3 .Three or more times

# SEX Character 1
# Sex
# 1 .Male
# 2 .Female

# SCHL Character 2
# Educational attainment
# bb .N/A (less than 3 years old)
# 01 .No schooling completed
# 02 .Nursery school, preschool
# 03 .Kindergarten
# 04 .Grade 1
# 05 .Grade 2
# 06 .Grade 3
# 07 .Grade 4
# 08 .Grade 5
# 09 .Grade 6
# 10 .Grade 7
# 11 .Grade 8
# 12 .Grade 9
# 13 .Grade 10
# 14 .Grade 11
# 15 .12th grade - no diploma
# 16 .Regular high school diploma
# 17 .GED or alternative credential
# 18 .Some college, but less than 1 year
# 19 .1 or more years of college credit, no degree
# 20 .Associate's degree
# 21 .Bachelor's degree
# 22 .Master's degree
# 23 .Professional degree beyond a bachelor's degree
# 24 .Doctorate degree

# OCCP Character 4
# Occupation recode based on 2010 OCC codes
# bbbb .N/A (less than 16 years old/NILF who last worked more than
# .5 years ago or never worked)
# 0010 .MGR-Chief Executives And Legislators
# 0020 .MGR-General And Operations Managers
# 0040 .MGR-Advertising And Promotions Managers
# 0050 .MGR-Marketing And Sales Managers
# 0060 .MGR-Public Relations And Fundraising Managers
# 0100 .MGR-Administrative Services Managers
# 0110 .MGR-Computer And Information Systems Managers
# 0120 .MGR-Financial Managers
# 0135 .MGR-Compensation And Benefits Managers
# 0136 .MGR-Human Resources Managers
# 0137 .MGR-Training And Development Managers
# 0140 .MGR-Industrial Production Managers
# 0150 .MGR-Purchasing Managers
# 0160 .MGR-Transportation, Storage, And Distribution Managers
# 0205 .MGR-Farmers, Ranchers, And Other Agricultural Managers
# 0220 .MGR-Construction Managers
# 0230 .MGR-Education Administrators
# 0300 .MGR-Architectural And Engineering Managers
# 0310 .MGR-Food Service Managers
# 0330 .MGR-Gaming Managers
# 0340 .MGR-Lodging Managers
# 0350 .MGR-Medical And Health Services Managers
# 0360 .MGR-Natural Sciences Managers
# 0410 .MGR-Property, Real Estate, And Community Association
# .Managers
# 0420 .MGR-Social And Community Service Managers
# 0425 .MGR-Emergency Management Directors
# 0430 .MGR-Miscellaneous Managers, Including Funeral Service
# .Managers And Postmasters And Mail Superintendents
# 0500 .BUS-Agents And Business Managers Of Artists, Performers,
# .And Athletes
# 0510 .BUS-Buyers And Purchasing Agents, Farm Products
# 0520 .BUS-Wholesale And Retail Buyers, Except Farm Products
# 0530 .BUS-Purchasing Agents, Except Wholesale, Retail, And Farm
# .Products
# 0540 .BUS-Claims Adjusters, Appraisers, Examiners, And
# .Investigators
# 0565 .BUS-Compliance Officers
# 0600 .BUS-Cost Estimators
# 0630 .BUS-Human Resources Workers
# 0640 .BUS-Compensation, Benefits, And Job Analysis Specialists
# 0650 .BUS-Training And Development Specialists
# 0700 .BUS-Logisticians
# 0710 .BUS-Management Analysts
# 0725 .BUS-Meeting, Convention, And Event Planners
# 86
# 0726 .BUS-Fundraisers
# 0735 .BUS-Market Research Analysts And Marketing Specialists
# 0740 .BUS-Business Operations Specialists, All Other
# 0800 .FIN-Accountants And Auditors
# 0810 .FIN-Appraisers And Assessors Of Real Estate
# 0820 .FIN-Budget Analysts
# 0830 .FIN-Credit Analysts
# 0840 .FIN-Financial Analysts
# 0850 .FIN-Personal Financial Advisors
# 0860 .FIN-Insurance Underwriters
# 0900 .FIN-Financial Examiners
# 0910 .FIN-Credit Counselors And Loan Officers
# 0930 .FIN-Tax Examiners And Collectors, And Revenue Agents
# 0940 .FIN-Tax Preparers
# 0950 .FIN-Financial Specialists, All Other
# 1005 .CMM-Computer And Information Research Scientists
# 1006 .CMM-Computer Systems Analysts
# 1007 .CMM-Information Security Analysts
# 1010 .CMM-Computer Programmers
# 1020 .CMM-Software Developers, Applications And Systems Software
# 1030 .CMM-Web Developers
# 1050 .CMM-Computer Support Specialists
# 1060 .CMM-Database Administrators
# 1105 .CMM-Network And Computer Systems Administrators
# 1106 .CMM-Computer Network Architects
# 1107 .CMM-Computer Occupations, All Other
# 1200 .CMM-Actuaries
# 1220 .CMM-Operations Research Analysts
# 1240 .CMM-Miscellaneous Mathematical Science Occupations,
# .Including Mathematicians And Statisticians
# 1300 .ENG-Architects, Except Naval
# 1310 .ENG-Surveyors, Cartographers, And Photogrammetrists
# 1320 .ENG-Aerospace Engineers
# 1340 .ENG-Biomedical And Agricultural Engineers
# 1350 .ENG-Chemical Engineers
# 1360 .ENG-Civil Engineers
# 1400 .ENG-Computer Hardware Engineers
# 1410 .ENG-Electrical And Electronics Engineers
# 1420 .ENG-Environmental Engineers
# 1430 .ENG-Industrial Engineers, Including Health And Safety
# 1440 .ENG-Marine Engineers And Naval Architects
# 1450 .ENG-Materials Engineers
# 1460 .ENG-Mechanical Engineers
# 1520 .ENG-Petroleum, Mining And Geological Engineers, Including
# .Mining Safety Engineers
# 1530 .ENG-Miscellaneous Engineers, Including Nuclear Engineers
# 1540 .ENG-Drafters
# 1550 .ENG-Engineering Technicians, Except Drafters
# 1560 .ENG-Surveying And Mapping Technicians
# 1600 .SCI-Agricultural And Food Scientists
# 1610 .SCI-Biological Scientists
# 1640 .SCI-Conservation Scientists And Foresters
# 1650 .SCI-Medical Scientists, And Life Scientists, All Other
# 1700 .SCI-Astronomers And Physicists
# 1710 .SCI-Atmospheric And Space Scientists
# 1720 .SCI-Chemists And Materials Scientists
# 1740 .SCI-Environmental Scientists And Geoscientists
# 87
# 1760 .SCI-Physical Scientists, All Other
# 1800 .SCI-Economists
# 1820 .SCI-Psychologists
# 1840 .SCI-Urban And Regional Planners
# 1860 .SCI-Miscellaneous Social Scientists, Including Survey
# .Researchers And Sociologists
# 1900 .SCI-Agricultural And Food Science Technicians
# 1910 .SCI-Biological Technicians
# 1920 .SCI-Chemical Technicians
# 1930 .SCI-Geological And Petroleum Technicians, And Nuclear
# .Technicians
# 1965 .SCI-Miscellaneous Life, Physical, And Social Science
# .Technicians, Including Social Science Research Assistants
# 2000 .CMS-Counselors
# 2010 .CMS-Social Workers
# 2015 .CMS-Probation Officers And Correctional Treatment
# .Specialists
# 2016 .CMS-Social And Human Service Assistants
# 2025 .CMS-Miscellaneous Community And Social Service Specialists,
# .Including Health Educators And Community Health Workers
# 2040 .CMS-Clergy
# 2050 .CMS-Directors, Religious Activities And Education
# 2060 .CMS-Religious Workers, All Other
# 2100 .LGL-Lawyers, And Judges, Magistrates, And Other Judicial
# .Workers
# 2105 .LGL-Judicial Law Clerks
# 2145 .LGL-Paralegals And Legal Assistants
# 2160 .LGL-Miscellaneous Legal Support Workers
# 2200 .EDU-Postsecondary Teachers
# 2300 .EDU-Preschool And Kindergarten Teachers
# 2310 .EDU-Elementary And Middle School Teachers
# 2320 .EDU-Secondary School Teachers
# 2330 .EDU-Special Education Teachers
# 2340 .EDU-Other Teachers And Instructors
# 2400 .EDU-Archivists, Curators, And Museum Technicians
# 2430 .EDU-Librarians
# 2440 .EDU-Library Technicians
# 2540 .EDU-Teacher Assistants
# 2550 .EDU-Other Education, Training, And Library Workers
# 2600 .ENT-Artists And Related Workers
# 2630 .ENT-Designers
# 2700 .ENT-Actors
# 2710 .ENT-Producers And Directors
# 2720 .ENT-Athletes, Coaches, Umpires, And Related Workers
# 2740 .ENT-Dancers And Choreographers
# 2750 .ENT-Musicians, Singers, And Related Workers
# 2760 .ENT-Entertainers And Performers, Sports And Related
# .Workers, All Other
# 2800 .ENT-Announcers
# 2810 .ENT-News Analysts, Reporters And Correspondents
# 2825 .ENT-Public Relations Specialists
# 2830 .ENT-Editors
# 2840 .ENT-Technical Writers
# 2850 .ENT-Writers And Authors
# 2860 .ENT-Miscellaneous Media And Communication Workers
# 88
# 2900 .ENT-Broadcast And Sound Engineering Technicians And Radio
# .Operators, And Media And Communication Equipment Workers,
# .All Other
# 2910 .ENT-Photographers
# 2920 .ENT-Television, Video, And Motion Picture Camera Operators
# .And Editors
# 3000 .MED-Chiropractors
# 3010 .MED-Dentists
# 3030 .MED-Dietitians And Nutritionists
# 3040 .MED-Optometrists
# 3050 .MED-Pharmacists
# 3060 .MED-Physicians And Surgeons
# 3110 .MED-Physician Assistants
# 3120 .MED-Podiatrists
# 3140 .MED-Audiologists
# 3150 .MED-Occupational Therapists
# 3160 .MED-Physical Therapists
# 3200 .MED-Radiation Therapists
# 3210 .MED-Recreational Therapists
# 3220 .MED-Respiratory Therapists
# 3230 .MED-Speech-Language Pathologists
# 3245 .MED-Other Therapists, Including Exercise Physiologists
# 3250 .MED-Veterinarians
# 3255 .MED-Registered Nurses
# 3256 .MED-Nurse Anesthetists
# 3258 .MED-Nurse Practitioners, And Nurse Midwives
# 3260 .MED-Health Diagnosing And Treating Practitioners, All Other
# 3300 .MED-Clinical Laboratory Technologists And Technicians
# 3310 .MED-Dental Hygienists
# 3320 .MED-Diagnostic Related Technologists And Technicians
# 3400 .MED-Emergency Medical Technicians And Paramedics
# 3420 .MED-Health Practitioner Support Technologists And
# .Technicians
# 3500 .MED-Licensed Practical And Licensed Vocational Nurses
# 3510 .MED-Medical Records And Health Information Technicians
# 3520 .MED-Opticians, Dispensing
# 3535 .MED-Miscellaneous Health Technologists And Technicians
# 3540 .MED-Other Healthcare Practitioners And Technical
# .Occupations
# 3600 .HLS-Nursing, Psychiatric, And Home Health Aides
# 3610 .HLS-Occupational Therapy Assistants And Aides
# 3620 .HLS-Physical Therapist Assistants And Aides
# 3630 .HLS-Massage Therapists
# 3640 .HLS-Dental Assistants
# 3645 .HLS-Medical Assistants
# 3646 .HLS-Medical Transcriptionists
# 3647 .HLS-Pharmacy Aides
# 3648 .HLS-Veterinary Assistants And Laboratory Animal Caretakers
# 3649 .HLS-Phlebotomists
# 3655 .HLS-Healthcare Support Workers, All Other, Including
# .Medical Equipment Preparers
# 3700 .PRT-First-Line Supervisors Of Correctional Officers
# 3710 .PRT-First-Line Supervisors Of Police And Detectives
# 3720 .PRT-First-Line Supervisors Of Fire Fighting And Prevention
# .Workers
# 3730 .PRT-First-Line Supervisors Of Protective Service Workers,
# .All Other
# 89
# 3740 .PRT-Firefighters
# 3750 .PRT-Fire Inspectors
# 3800 .PRT-Bailiffs, Correctional Officers, And Jailers
# 3820 .PRT-Detectives And Criminal Investigators
# 3840 .PRT-Miscellaneous Law Enforcement Workers
# 3850 .PRT-Police Officers
# 3900 .PRT-Animal Control Workers
# 3910 .PRT-Private Detectives And Investigators
# 3930 .PRT-Security Guards And Gaming Surveillance Officers
# 3940 .PRT-Crossing Guards
# 3945 .PRT-Transportation Security Screeners
# 3955 .PRT-Lifeguards And Other Recreational, And All Other
# .Protective Service Workers
# 4000 .EAT-Chefs And Head Cooks
# 4010 .EAT-First-Line Supervisors Of Food Preparation And Serving
# .Workers
# 4020 .EAT-Cooks
# 4030 .EAT-Food Preparation Workers
# 4040 .EAT-Bartenders
# 4050 .EAT-Combined Food Preparation And Serving Workers,
# .Including Fast Food
# 4060 .EAT-Counter Attendants, Cafeteria, Food Concession, And
# .Coffee Shop
# 4110 .EAT-Waiters And Waitresses
# 4120 .EAT-Food Servers, Nonrestaurant
# 4130 .EAT-Miscellaneous Food Preparation And Serving Related
# .Workers, Including Dining Room And Cafeteria Attendants And
# .Bartender Helpers
# 4140 .EAT-Dishwashers
# 4150 .EAT-Hosts And Hostesses, Restaurant, Lounge, And Coffee
# .Shop
# 4200 .CLN-First-Line Supervisors Of Housekeeping And Janitorial
# .Workers
# 4210 .CLN-First-Line Supervisors Of Landscaping, Lawn Service,
# .And Groundskeeping Workers
# 4220 .CLN-Janitors And Building Cleaners
# 4230 .CLN-Maids And Housekeeping Cleaners
# 4240 .CLN-Pest Control Workers
# 4250 .CLN-Grounds Maintenance Workers
# 4300 .PRS-First-Line Supervisors Of Gaming Workers
# 4320 .PRS-First-Line Supervisors Of Personal Service Workers
# 4340 .PRS-Animal Trainers
# 4350 .PRS-Nonfarm Animal Caretakers
# 4400 .PRS-Gaming Services Workers
# 4410 .PRS-Motion Picture Projectionists
# 4420 .PRS-Ushers, Lobby Attendants, And Ticket Takers
# 4430 .PRS-Miscellaneous Entertainment Attendants And Related
# .Workers
# 4460 .PRS-Embalmers And Funeral Attendants
# 4465 .PRS-Morticians, Undertakers, And Funeral Directors
# 4500 .PRS-Barbers
# 4510 .PRS-Hairdressers, Hairstylists, And Cosmetologists
# 4520 .PRS-Miscellaneous Personal Appearance Workers
# 4530 .PRS-Baggage Porters, Bellhops, And Concierges
# 4540 .PRS-Tour And Travel Guides
# 4600 .PRS-Childcare Workers
# 4610 .PRS-Personal Care Aides
# 90
# 4620 .PRS-Recreation And Fitness Workers
# 4640 .PRS-Residential Advisors
# 4650 .PRS-Personal Care And Service Workers, All Other
# 4700 .SAL-First-Line Supervisors Of Retail Sales Workers
# 4710 .SAL-First-Line Supervisors Of Non-Retail Sales Workers
# 4720 .SAL-Cashiers
# 4740 .SAL-Counter And Rental Clerks
# 4750 .SAL-Parts Salespersons
# 4760 .SAL-Retail Salespersons
# 4800 .SAL-Advertising Sales Agents
# 4810 .SAL-Insurance Sales Agents
# 4820 .SAL-Securities, Commodities, And Financial Services Sales
# .Agents
# 4830 .SAL-Travel Agents
# 4840 .SAL-Sales Representatives, Services, All Other
# 4850 .SAL-Sales Representatives, Wholesale And Manufacturing
# 4900 .SAL-Models, Demonstrators, And Product Promoters
# 4920 .SAL-Real Estate Brokers And Sales Agents
# 4930 .SAL-Sales Engineers
# 4940 .SAL-Telemarketers
# 4950 .SAL-Door-To-Door Sales Workers, News And Street Vendors,
# .And Related Workers
# 4965 .SAL-Sales And Related Workers, All Other
# 5000 .OFF-First-Line Supervisors Of Office And Administrative
# .Support Workers
# 5010 .OFF-Switchboard Operators, Including Answering Service
# 5020 .OFF-Telephone Operators
# 5030 .OFF-Communications Equipment Operators, All Other
# 5100 .OFF-Bill And Account Collectors
# 5110 .OFF-Billing And Posting Clerks
# 5120 .OFF-Bookkeeping, Accounting, And Auditing Clerks
# 5130 .OFF-Gaming Cage Workers
# 5140 .OFF-Payroll And Timekeeping Clerks
# 5150 .OFF-Procurement Clerks
# 5160 .OFF-Tellers
# 5165 .OFF-Financial Clerks, All Other
# 5200 .OFF-Brokerage Clerks
# 5220 .OFF-Court, Municipal, And License Clerks
# 5230 .OFF-Credit Authorizers, Checkers, And Clerks
# 5240 .OFF-Customer Service Representatives
# 5250 .OFF-Eligibility Interviewers, Government Programs
# 5260 .OFF-File Clerks
# 5300 .OFF-Hotel, Motel, And Resort Desk Clerks
# 5310 .OFF-Interviewers, Except Eligibility And Loan
# 5320 .OFF-Library Assistants, Clerical
# 5330 .OFF-Loan Interviewers And Clerks
# 5340 .OFF-New Accounts Clerks
# 5350 .OFF-Correspondence Clerks And Order Clerks
# 5360 .OFF-Human Resources Assistants, Except Payroll And
# .Timekeeping
# 5400 .OFF-Receptionists And Information Clerks
# 5410 .OFF-Reservation And Transportation Ticket Agents And Travel
# .Clerks
# 5420 .OFF-Information And Record Clerks, All Other
# 5500 .OFF-Cargo And Freight Agents
# 5510 .OFF-Couriers And Messengers
# 5520 .OFF-Dispatchers
# 91
# 5530 .OFF-Meter Readers, Utilities
# 5540 .OFF-Postal Service Clerks
# 5550 .OFF-Postal Service Mail Carriers
# 5560 .OFF-Postal Service Mail Sorters, Processors, And Processing
# .Machine Operators
# 5600 .OFF-Production, Planning, And Expediting Clerks
# 5610 .OFF-Shipping, Receiving, And Traffic Clerks
# 5620 .OFF-Stock Clerks And Order Fillers
# 5630 .OFF-Weighers, Measurers, Checkers, And Samplers,
# .Recordkeeping
# 5700 .OFF-Secretaries And Administrative Assistants
# 5800 .OFF-Computer Operators
# 5810 .OFF-Data Entry Keyers
# 5820 .OFF-Word Processors And Typists
# 5840 .OFF-Insurance Claims And Policy Processing Clerks
# 5850 .OFF-Mail Clerks And Mail Machine Operators, Except Postal
# .Service
# 5860 .OFF-Office Clerks, General
# 5900 .OFF-Office Machine Operators, Except Computer
# 5910 .OFF-Proofreaders And Copy Markers
# 5920 .OFF-Statistical Assistants
# 5940 .OFF-Miscellaneous Office And Administrative Support
# .Workers, Including Desktop Publishers
# 6005 .FFF-First-Line Supervisors Of Farming, Fishing, And
# .Forestry Workers
# 6010 .FFF-Agricultural Inspectors
# 6040 .FFF-Graders And Sorters, Agricultural Products
# 6050 .FFF-Miscellaneous Agricultural Workers, Including Animal
# .Breeders
# 6100 .FFF-Fishing And Hunting Workers
# 6120 .FFF-Forest And Conservation Workers
# 6130 .FFF-Logging Workers
# 6200 .CON-First-Line Supervisors Of Construction Trades And
# .Extraction Workers
# 6210 .CON-Boilermakers
# 6220 .CON-Brickmasons, Blockmasons, Stonemasons, And Reinforcing
# .Iron And Rebar Workers
# 6230 .CON-Carpenters
# 6240 .CON-Carpet, Floor, And Tile Installers And Finishers
# 6250 .CON-Cement Masons, Concrete Finishers, And Terrazzo Workers
# 6260 .CON-Construction Laborers
# 6300 .CON-Paving, Surfacing, And Tamping Equipment Operators
# 6320 .CON-Construction Equipment Operators, Except Paving,
# .Surfacing, And Tamping Equipment Operators
# 6330 .CON-Drywall Installers, Ceiling Tile Installers, And Tapers
# 6355 .CON-Electricians
# 6360 .CON-Glaziers
# 6400 .CON-Insulation Workers
# 6420 .CON-Painters And Paperhangers
# 6440 .CON-Pipelayers, Plumbers, Pipefitters, And Steamfitters
# 6460 .CON-Plasterers And Stucco Masons
# 6515 .CON-Roofers
# 6520 .CON-Sheet Metal Workers
# 6530 .CON-Structural Iron And Steel Workers
# 6600 .CON-Helpers, Construction Trades
# 6660 .CON-Construction And Building Inspectors
# 6700 .CON-Elevator Installers And Repairers
# 92
# 6710 .CON-Fence Erectors
# 6720 .CON-Hazardous Materials Removal Workers
# 6730 .CON-Highway Maintenance Workers
# 6740 .CON-Rail-Track Laying And Maintenance Equipment Operators
# 6765 .CON-Miscellaneous Construction Workers, Including Solar
# .Photovoltaic Installers, Septic Tank Servicers And Sewer
# .Pipe Cleaners
# 6800 .EXT-Derrick, Rotary Drill, And Service Unit Operators, And
# .Roustabouts, Oil, Gas, And Mining
# 6820 .EXT-Earth Drillers, Except Oil And Gas
# 6830 .EXT-Explosives Workers, Ordnance Handling Experts, And
# .Blasters
# 6840 .EXT-Mining Machine Operators
# 6940 .EXT-Miscellaneous Extraction Workers, Including Roof
# .Bolters And Helpers
# 7000 .RPR-First-Line Supervisors Of Mechanics, Installers, And
# .Repairers
# 7010 .RPR-Computer, Automated Teller, And Office Machine
# .Repairers
# 7020 .RPR-Radio And Telecommunications Equipment Installers And
# .Repairers
# 7030 .RPR-Avionics Technicians
# 7040 .RPR-Electric Motor, Power Tool, And Related Repairers
# 7100 .RPR-Electrical And Electronics Repairers, Transportation
# .Equipment, And Industrial And Utility
# 7110 .RPR-Electronic Equipment Installers And Repairers, Motor
# .Vehicles
# 7120 .RPR-Electronic Home Entertainment Equipment Installers And
# .Repairers
# 7130 .RPR-Security And Fire Alarm Systems Installers
# 7140 .RPR-Aircraft Mechanics And Service Technicians
# 7150 .RPR-Automotive Body And Related Repairers
# 7160 .RPR-Automotive Glass Installers And Repairers
# 7200 .RPR-Automotive Service Technicians And Mechanics
# 7210 .RPR-Bus And Truck Mechanics And Diesel Engine Specialists
# 7220 .RPR-Heavy Vehicle And Mobile Equipment Service Technicians
# .And Mechanics
# 7240 .RPR-Small Engine Mechanics
# 7260 .RPR-Miscellaneous Vehicle And Mobile Equipment Mechanics,
# .Installers, And Repairers
# 7300 .RPR-Control And Valve Installers And Repairers
# 7315 .RPR-Heating, Air Conditioning, And Refrigeration Mechanics
# .And Installers
# 7320 .RPR-Home Appliance Repairers
# 7330 .RPR-Industrial And Refractory Machinery Mechanics
# 7340 .RPR-Maintenance And Repair Workers, General
# 7350 .RPR-Maintenance Workers, Machinery
# 7360 .RPR-Millwrights
# 7410 .RPR-Electrical Power-Line Installers And Repairers
# 7420 .RPR-Telecommunications Line Installers And Repairers
# 7430 .RPR-Precision Instrument And Equipment Repairers
# 7510 .RPR-Coin, Vending, And Amusement Machine Servicers And
# .Repairers
# 7540 .RPR-Locksmiths And Safe Repairers
# 7560 .RPR-Riggers
# 7610 .RPR-Helpers-Installation, Maintenance, And Repair Workers
# 93
# 7630 .RPR-Miscellaneous Installation, Maintenance, And Repair
# .Workers, Including Wind Turbine Service Technicians
# 7700 .PRD-First-Line Supervisors Of Production And Operating
# .Workers
# 7710 .PRD-Aircraft Structure, Surfaces, Rigging, And Systems
# .Assemblers
# 7720 .PRD-Electrical, Electronics, And Electromechanical
# .Assemblers
# 7730 .PRD-Engine And Other Machine Assemblers
# 7740 .PRD-Structural Metal Fabricators And Fitters
# 7750 .PRD-Miscellaneous Assemblers And Fabricators
# 7800 .PRD-Bakers
# 7810 .PRD-Butchers And Other Meat, Poultry, And Fish Processing
# .Workers
# 7830 .PRD-Food And Tobacco Roasting, Baking, And Drying Machine
# .Operators And Tenders
# 7840 .PRD-Food Batchmakers
# 7850 .PRD-Food Cooking Machine Operators And Tenders
# 7855 .PRD-Food Processing Workers, All Other
# 7900 .PRD-Computer Control Programmers And Operators
# 7920 .PRD-Extruding And Drawing Machine Setters, Operators, And
# .Tenders, Metal And Plastic
# 7930 .PRD-Forging Machine Setters, Operators, And Tenders, Metal
# .And Plastic
# 7940 .PRD-Rolling Machine Setters, Operators, And Tenders, Metal
# .And Plastic
# 7950 .PRD-Machine Tool Cutting Setters, Operators, And Tenders,
# .Metal And Plastic
# 8030 .PRD-Machinists
# 8040 .PRD-Metal Furnace Operators, Tenders, Pourers, And Casters
# 8100 .PRD-Model Makers, Patternmakers, And Molding Machine
# .Setters, Metal And Plastic
# 8130 .PRD-Tool And Die Makers
# 8140 .PRD-Welding, Soldering, And Brazing Workers
# 8220 .PRD-Miscellaneous Metal Workers And Plastic Workers,
# .Including Multiple Machine Tool Setters
# 8250 .PRD-Prepress Technicians And Workers
# 8255 .PRD-Printing Press Operators
# 8256 .PRD-Print Binding And Finishing Workers
# 8300 .PRD-Laundry And Dry-Cleaning Workers
# 8310 .PRD-Pressers, Textile, Garment, And Related Materials
# 8320 .PRD-Sewing Machine Operators
# 8330 .PRD-Shoe And Leather Workers
# 8350 .PRD-Tailors, Dressmakers, And Sewers
# 8400 .PRD-Textile Bleaching And Dyeing, And Cutting Machine
# .Setters, Operators, And Tenders
# 8410 .PRD-Textile Knitting And Weaving Machine Setters,
# .Operators, And Tenders
# 8420 .PRD-Textile Winding, Twisting, And Drawing Out Machine
# .Setters, Operators, And Tenders
# 8450 .PRD-Upholsterers
# 8460 .PRD-Miscellaneous Textile, Apparel, And Furnishings
# .Workers, Except Upholsterers
# 8500 .PRD-Cabinetmakers And Bench Carpenters
# 8510 .PRD-Furniture Finishers
# 8530 .PRD-Sawing Machine Setters, Operators, And Tenders, Wood
# 94
# 8540 .PRD-Woodworking Machine Setters, Operators, And Tenders,
# .Except Sawing
# 8550 .PRD-Miscellaneous Woodworkers, Including Model Makers And
# .Patternmakers
# 8600 .PRD-Power Plant Operators, Distributors, And Dispatchers
# 8610 .PRD-Stationary Engineers And Boiler Operators
# 8620 .PRD-Water And Wastewater Treatment Plant And System
# .Operators
# 8630 .PRD-Miscellaneous Plant And System Operators
# 8640 .PRD-Chemical Processing Machine Setters, Operators, And
# .Tenders
# 8650 .PRD-Crushing, Grinding, Polishing, Mixing, And Blending
# .Workers
# 8710 .PRD-Cutting Workers
# 8720 .PRD-Extruding, Forming, Pressing, And Compacting Machine
# .Setters, Operators, And Tenders
# 8730 .PRD-Furnace, Kiln, Oven, Drier, And Kettle Operators And
# .Tenders
# 8740 .PRD-Inspectors, Testers, Sorters, Samplers, And Weighers
# 8750 .PRD-Jewelers And Precious Stone And Metal Workers
# 8760 .PRD-Medical, Dental, And Ophthalmic Laboratory Technicians
# 8800 .PRD-Packaging And Filling Machine Operators And Tenders
# 8810 .PRD-Painting Workers
# 8830 .PRD-Photographic Process Workers And Processing Machine
# .Operators
# 8850 .PRD-Adhesive Bonding Machine Operators And Tenders
# 8910 .PRD-Etchers And Engravers
# 8920 .PRD-Molders, Shapers, And Casters, Except Metal And Plastic
# 8930 .PRD-Paper Goods Machine Setters, Operators, And Tenders
# 8940 .PRD-Tire Builders
# 8950 .PRD-Helpers-Production Workers
# 8965 .PRD-Miscellaneous Production Workers, Including
# .Semiconductor Processors
# 9000 .TRN-Supervisors Of Transportation And Material Moving
# .Workers
# 9030 .TRN-Aircraft Pilots And Flight Engineers
# 9040 .TRN-Air Traffic Controllers And Airfield Operations
# .Specialists
# 9050 .TRN-Flight Attendants
# 9110 .TRN-Ambulance Drivers And Attendants, Except Emergency
# .Medical Technicians
# 9120 .TRN-Bus Drivers
# 9130 .TRN-Driver/Sales Workers And Truck Drivers
# 9140 .TRN-Taxi Drivers And Chauffeurs
# 9150 .TRN-Motor Vehicle Operators, All Other
# 9200 .TRN-Locomotive Engineers And Operators
# 9240 .TRN-Railroad Conductors And Yardmasters
# 9260 .TRN-Subway, Streetcar, And Other Rail Transportation
# .Workers
# 9300 .TRN-Sailors And Marine Oilers, And Ship Engineers
# 9310 .TRN-Ship And Boat Captains And Operators
# 9350 .TRN-Parking Lot Attendants
# 9360 .TRN-Automotive And Watercraft Service Attendants
# 9410 .TRN-Transportation Inspectors
# 9415 .TRN-Transportation Attendants, Except Flight Attendants
# 9420 .TRN-Miscellaneous Transportation Workers, Including Bridge
# .And Lock Tenders And Traffic Technicians
# 95
# 9510 .TRN-Crane And Tower Operators
# 9520 .TRN-Dredge, Excavating, And Loading Machine Operators
# 9560 .TRN-Conveyor Operators And Tenders, And Hoist And Winch
# .Operators
# 9600 .TRN-Industrial Truck And Tractor Operators
# 9610 .TRN-Cleaners Of Vehicles And Equipment
# 9620 .TRN-Laborers And Freight, Stock, And Material Movers, Hand
# 9630 .TRN-Machine Feeders And Offbearers
# 9640 .TRN-Packers And Packagers, Hand
# 9650 .TRN-Pumping Station Operators
# 9720 .TRN-Refuse And Recyclable Material Collectors
# 9750 .TRN-Miscellaneous Material Moving Workers, Including Mine
# .Shuttle Car Operators, And Tank Car, Truck, And Ship
# .Loaders
# 9800 .MIL-Military Officer Special And Tactical Operations
# .Leaders
# 9810 .MIL-First-Line Enlisted Military Supervisors
# 9820 .MIL-Military Enlisted Tactical Operations And Air/Weapons
# .Specialists And Crew Members
# 9830 .MIL-Military, Rank Not Specified
# 9920 .Unemployed And Last Worked 5 Years Ago Or Earlier Or Never
# .Worked

# PINCP Numeric 7
# Total person's income (signed, use ADJINC to adjust to constant
# dollars)
# bbbbbbb .N/A (less than 15 years old)
# 0000000 .None
# -019998 .Loss of $19999 or more (Rounded & bottom-coded
# .components)
# -000001..-019997 .Loss $1 to $19997 (Rounded components)
# 0000001 .$1 or break even
# 0000002..4209995 .$2 to $4209995(Rounded & top-coded components)
