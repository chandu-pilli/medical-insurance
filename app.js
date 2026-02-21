/**
 * Omegle Health - Health Insurance Cost Prediction System
    * ML - powered insurance cost prediction with Python backend
        */

// ============================================
// API Configuration
// ============================================
const API_BASE_URL = 'http://localhost:5000';
let backendAvailable = false;
let lastApiPlans = null; // Cache last API response

// Check if backend is available
async function checkBackend() {
    try {
        const res = await fetch(`${API_BASE_URL}/api/health`, { signal: AbortSignal.timeout(2000) });
        const data = await res.json();
        backendAvailable = data.status === 'ok' && data.model_loaded;
        if (backendAvailable) {
            console.log('🧠 ML Backend connected! Model R²:', data.metrics?.test_r2?.toFixed(4));
        }
    } catch (e) {
        backendAvailable = false;
        console.log('⚠️ ML Backend not available, using client-side prediction');
    }
}
checkBackend();

// ============================================
// State Management
// ============================================
const AppState = {
    currentStep: 0,
    purpose: '',
    pincode: '',
    location: '',
    mobile: '',
    generatedOtp: '', // Store the OTP for verification
    fullName: '',
    email: '',
    gender: 'male',
    selectedMembers: [],
    memberDetails: [],  // { type, label, dob, age }
    conditions: [],
    isSmoker: false,
    bmi: null,
    annualIncome: '',
    sumInsured: 10,  // in Lacs
    tenure: 1,
    selectedPlan: 0,
    sonCount: 1,
    daughterCount: 1,
};

// ============================================
// PIN Code Database (sample data)
// ============================================
const PIN_DATABASE = {
    // Andhra Pradesh Districts
    '530001': 'VISAKHAPATNAM, ANDHRA PRADESH',
    '530002': 'VISAKHAPATNAM (CITY), ANDHRA PRADESH',
    '530016': 'VISAKHAPATNAM (STEEL PLANT), ANDHRA PRADESH',
    '520001': 'VIJAYAWADA (KRISHNA), ANDHRA PRADESH',
    '520002': 'VIJAYAWADA (BUDHAVARAPUPETA), ANDHRA PRADESH',
    '522001': 'GUNTUR (CITY), ANDHRA PRADESH',
    '522002': 'GUNTUR (OFFICE), ANDHRA PRADESH',
    '524001': 'NELLORE (CITY), ANDHRA PRADESH',
    '524002': 'NELLORE (SANTHAPETA), ANDHRA PRADESH',
    '518001': 'KURNOOL (CITY), ANDHRA PRADESH',
    '518002': 'KURNOOL (NEW TOWN), ANDHRA PRADESH',
    '517501': 'TIRUPATI (CITTY), ANDHRA PRADESH',
    '517507': 'TIRUPATI (TIRUMALA), ANDHRA PRADESH',
    '533001': 'KAKINADA (EAST GODAVARI), ANDHRA PRADESH',
    '533003': 'KAKINADA (CITY), ANDHRA PRADESH',
    '533101': 'RAJAHMUNDRY (EAST GODAVARI), ANDHRA PRADESH',
    '534001': 'ELURU (WEST GODAVARI), ANDHRA PRADESH',
    '534201': 'BHIMAVARAM (WEST GODAVARI), ANDHRA PRADESH',
    '515001': 'ANANTAPUR (CITY), ANDHRA PRADESH',
    '515002': 'ANANTAPUR (NEW TOWN), ANDHRA PRADESH',
    '516001': 'KADAPA (CITY), ANDHRA PRADESH',
    '535001': 'VIZIANAGARAM (CITY), ANDHRA PRADESH',
    '535002': 'VIZIANAGARAM (NEW TOWN), ANDHRA PRADESH',
    '532001': 'SRIKAKULAM (CITY), ANDHRA PRADESH',
    '523001': 'ONGOLE (PRAKASAM), ANDHRA PRADESH',
    '533221': 'AMALAPURAM (EAST GODAVARI), ANDHRA PRADESH',
    '518501': 'NANDYAL (CITY), ANDHRA PRADESH',
    '521001': 'MACHILIPATNAM (KRISHNA), ANDHRA PRADESH',
    '517001': 'CHITTOOR (CITY), ANDHRA PRADESH',

    // Metro Cities
    '110001': 'NEW DELHI, DELHI',
    '400001': 'MUMBAI, MAHARASHTRA',
    '500001': 'HYDERABAD (CITY), TELANGANA',
    '500081': 'HITEC CITY, TELANGANA',
    '600001': 'CHENNAI, TAMIL NADU',
    '560001': 'BENGALURU, KARNATAKA',
    '700001': 'KOLKATA, WEST BENGAL',
    '302001': 'JAIPUR, RAJASTHAN',
    '380001': 'AHMEDABAD, GUJARAT',
    '411001': 'PUNE, MAHARASHTRA',
    '226001': 'LUCKNOW, UTTAR PRADESH',
    '122001': 'GURUGRAM, HARYANA',
    '201301': 'NOIDA, UTTAR PRADESH',
};

// ============================================
// Insurance Cost Prediction Engine
// ============================================
class InsurancePredictionEngine {
    constructor() {
        // Base rates per lakh of sum insured (annual, per member category)
        this.baseRates = {
            self: { child: 800, young: 1800, adult: 2800, middle: 4500, senior: 8500 },
            wife: { child: 800, young: 1700, adult: 2600, middle: 4200, senior: 8200 },
            husband: { child: 800, young: 1900, adult: 2900, middle: 4600, senior: 8600 },
            son: { child: 600, young: 1200, adult: 2200, middle: 3800, senior: 7500 },
            daughter: { child: 600, young: 1200, adult: 2200, middle: 3800, senior: 7500 },
            father: { child: 800, young: 2000, adult: 3200, middle: 5500, senior: 10500 },
            mother: { child: 800, young: 1900, adult: 3000, middle: 5200, senior: 10000 },
            fatherInLaw: { child: 800, young: 2000, adult: 3200, middle: 5500, senior: 10500 },
            motherInLaw: { child: 800, young: 1900, adult: 3000, middle: 5200, senior: 10000 },
        };

        // Condition multipliers
        this.conditionMultipliers = {
            diabetes: 1.25,
            hypertension: 1.20,
            thyroid: 1.10,
            asthma: 1.15,
            heart: 1.40,
            none: 1.0,
        };

        // BMI multipliers
        this.bmiMultipliers = [
            { range: [0, 18.5], mult: 1.10 },    // Underweight
            { range: [18.5, 25], mult: 1.00 },    // Normal
            { range: [25, 30], mult: 1.15 },       // Overweight
            { range: [30, 35], mult: 1.25 },       // Obese class 1
            { range: [35, 100], mult: 1.35 },      // Obese class 2+
        ];

        // Region multipliers based on metro tier
        this.regionMultipliers = {
            'DELHI': 1.15,
            'MAHARASHTRA': 1.12,
            'KARNATAKA': 1.10,
            'TELANGANA': 1.08,
            'TAMIL NADU': 1.08,
            'WEST BENGAL': 1.05,
            'KERALA': 1.06,
            'HARYANA': 1.10,
            'UTTAR PRADESH': 1.02,
            'RAJASTHAN': 1.00,
            'GUJARAT': 1.03,
            'ANDHRA PRADESH': 1.02,
            'PUNJAB': 1.04,
            'CHANDIGARH': 1.10,
            'BIHAR': 0.98,
            'ODISHA': 0.98,
            'JHARKHAND': 0.98,
            'MADHYA PRADESH': 1.00,
        };

        // Tenure discounts
        this.tenureDiscounts = {
            1: 1.0,
            2: 0.92,  // 8% off
            3: 0.85,  // 15% off
        };

        // Plan tiers
        this.planTiers = [
            {
                name: 'MediShield Essential',
                multiplier: 0.75,
                discount: 32,
                features: [
                    'Basic hospitalization cover',
                    'Annual Health Checkup',
                    'Consumables covered',
                    'Day care procedures',
                    'Ambulance charges',
                ],
                description: 'Comprehensive plan for essential coverage',
            },
            {
                name: 'MediShield Premier',
                multiplier: 1.0,
                discount: 27,
                recommended: true,
                features: [
                    'Comprehensive plan + Additional Riders',
                    'Health Checkup',
                    'Consumables covered',
                    '100% hospital bills paid* - No co-payment or room rent capping',
                    'Restoration benefit',
                    'AYUSH treatment covered',
                ],
                description: 'Comprehensive plan + Additional Riders',
            },
            {
                name: 'MediShield Premier Plus',
                multiplier: 1.3,
                discount: 25,
                features: [
                    '60L = 10L MediShield Premier + 50L MediShield Plus',
                    'Health Checkup',
                    'Consumables covered',
                    '100% hospital bills paid* - No co-payment or room rent capping',
                    'Global coverage',
                    'Air ambulance',
                    'Organ donor expenses',
                ],
                description: '60L coverage with super top-up',
            },
        ];
    }

    getAgeCategory(age) {
        if (age <= 17) return 'child';
        if (age <= 30) return 'young';
        if (age <= 45) return 'adult';
        if (age <= 60) return 'middle';
        return 'senior';
    }

    getRegionMultiplier(location) {
        if (!location) return 1.0;
        const state = location.split(',').pop()?.trim().toUpperCase() || '';
        return this.regionMultipliers[state] || 1.0;
    }

    getBmiMultiplier(bmi) {
        if (!bmi || bmi <= 0) return 1.0;
        for (const { range, mult } of this.bmiMultipliers) {
            if (bmi >= range[0] && bmi < range[1]) return mult;
        }
        return 1.0;
    }

    predict(state) {
        const { memberDetails, conditions, isSmoker, bmi, sumInsured, tenure, location } = state;

        // Calculate total base premium for all members
        let totalBase = 0;
        for (const member of memberDetails) {
            const category = this.getAgeCategory(member.age);
            const memberType = member.type || 'self';
            const rates = this.baseRates[memberType] || this.baseRates.self;
            const baseRate = rates[category] || 2500;
            totalBase += baseRate * sumInsured;
        }

        // Apply condition multipliers (compound)
        let conditionMult = 1.0;
        if (conditions.length > 0 && !conditions.includes('none')) {
            for (const cond of conditions) {
                conditionMult *= (this.conditionMultipliers[cond] || 1.0);
            }
        }

        // Smoker surcharge
        const smokerMult = isSmoker ? 1.20 : 1.0;

        // BMI multiplier
        const bmiMult = this.getBmiMultiplier(bmi);

        // Region multiplier
        const regionMult = this.getRegionMultiplier(location);

        // Family discount (more members = slight discount)
        const familyDiscount = memberDetails.length >= 4 ? 0.92 :
            memberDetails.length >= 3 ? 0.95 :
                memberDetails.length >= 2 ? 0.97 : 1.0;

        // Tenure discount
        const tenureDisc = this.tenureDiscounts[tenure] || 1.0;

        // Calculate final base premium
        const basePremium = totalBase * conditionMult * smokerMult * bmiMult * regionMult * familyDiscount * tenureDisc * tenure;

        // Generate plans
        const plans = this.planTiers.map(tier => {
            const premium = Math.round(basePremium * tier.multiplier);
            const originalPrice = Math.round(premium / (1 - tier.discount / 100));
            return {
                name: tier.name,
                premium,
                originalPrice,
                discount: tier.discount,
                features: tier.features,
                description: tier.description,
                recommended: tier.recommended || false,
            };
        });

        return plans;
    }
}

const predictor = new InsurancePredictionEngine();

// ============================================
// DOM Elements
// ============================================
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ============================================
// Background Particles
// ============================================
function createParticles() {
    const container = $('#bgParticles');
    const colors = ['#212121', '#424242', '#90e0ef', '#caf0f8', '#0077b6'];

    for (let i = 0; i < 20; i++) {
        const particle = document.createElement('div');
        particle.classList.add('bg-particle');
        const size = Math.random() * 12 + 4;
        const color = colors[Math.floor(Math.random() * colors.length)];
        particle.style.cssText = `
            width: ${size}px;
            height: ${size}px;
            background: ${color};
            left: ${Math.random() * 100}%;
            animation-duration: ${Math.random() * 15 + 15}s;
            animation-delay: ${Math.random() * 10}s;
        `;
        container.appendChild(particle);
    }
}

// ============================================
// Navigation & Step Management
// ============================================
function goToStep(step, reverse = false) {
    const oldStep = AppState.currentStep;
    AppState.currentStep = step;

    // Hide all steps
    $$('.step-card').forEach(card => card.classList.remove('active'));

    // Show current step
    const currentCard = $(`#step${step}`);
    if (currentCard) {
        currentCard.classList.add('active');
        currentCard.style.animation = reverse ? 'slideInReverse 0.45s cubic-bezier(0.34, 1.56, 0.64, 1)' : 'slideIn 0.45s cubic-bezier(0.34, 1.56, 0.64, 1)';
    }

    // Update progress
    updateProgress(step);

    // Update header subtitle
    updateHeaderSubtitle(step);

    // Hide progress & promo on landing page and results
    if (step === 0 || step === 7) {
        $('#progressContainer').style.display = 'none';
        $('#promoBanner').style.display = 'none';
        $('#disclaimer').style.display = 'none';
    } else {
        $('#progressContainer').style.display = '';
        if (step !== 0) $('#promoBanner').style.display = '';
        if (step !== 0) $('#disclaimer').style.display = '';
    }

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function updateProgress(step) {
    const actualStep = Math.min(step, 6);
    const pct = (actualStep / 6) * 100;
    $('#progressFill').style.width = `${pct}%`;

    $$('.progress-step').forEach((el, i) => {
        const s = i + 1;
        el.classList.remove('active', 'completed');
        if (s < actualStep) el.classList.add('completed');
        if (s === actualStep) el.classList.add('active');
    });
}

function updateHeaderSubtitle(step) {
    const messages = {
        0: "Hi! I'm Mahee😊, your AI insurance advisor. Let me help you find the perfect plan!",
        1: "What would you like to do today?",
        2: "Where do you live? This helps us find localized plans.",
        3: "Who should we address the policy to?",
        4: "Your mobile number is required to send the official quote.",
        5: "Who all would you like to insure?",
        6: "Almost there! Tell us a bit about everyone's age.",
        7: "Here are your personalized insurance plans! 🎉",
    };
    $('#headerSubtitle').textContent = messages[step] || messages[0];
}

// ============================================
// Step 0: Landing Page
// ============================================
$('#startExploringBtn').addEventListener('click', () => {
    goToStep(1);
});

// ============================================
// Step 1: Purpose
// ============================================
$$('.purpose-card').forEach(btn => {
    btn.addEventListener('click', () => {
        AppState.purpose = btn.dataset.purpose;

        // Animate click
        btn.style.transform = 'scale(0.97)';
        setTimeout(() => {
            btn.style.transform = '';
            goToStep(2);
        }, 150);
    });
});

// ============================================
// Step 2: Location & Contact
// ============================================
const pincodeInput = $('#pincode');
const locationDisplay = $('#locationDisplay');
const pincodeStatus = $('#pincodeStatus');

pincodeInput.addEventListener('input', (e) => {
    const val = e.target.value.replace(/\D/g, '');
    e.target.value = val;

    if (val.length === 6) {
        const loc = PIN_DATABASE[val];
        if (loc) {
            AppState.pincode = val;
            AppState.location = loc;
            locationDisplay.textContent = loc;
            pincodeInput.classList.add('valid');
            pincodeInput.classList.remove('invalid');
            pincodeStatus.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" fill="#45b7af"/><polyline points="8 12 11 15 16 9" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/></svg>';
            pincodeStatus.classList.add('show');
        } else {
            // Auto-generate location for unknown pins
            const state = getStateFromPin(val);
            AppState.pincode = val;
            AppState.location = state;
            locationDisplay.textContent = state;
            pincodeInput.classList.add('valid');
            pincodeInput.classList.remove('invalid');
            pincodeStatus.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" fill="#45b7af"/><polyline points="8 12 11 15 16 9" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/></svg>';
            pincodeStatus.classList.add('show');
        }
    } else {
        locationDisplay.textContent = '';
        pincodeInput.classList.remove('valid', 'invalid');
        pincodeStatus.classList.remove('show');
        AppState.location = '';
    }
});

function getStateFromPin(pin) {
    const first2 = parseInt(pin.substring(0, 2));
    const first3 = parseInt(pin.substring(0, 3));

    // Refined Andhra Pradesh District Prediction
    if (first2 >= 51 && first2 <= 53) {
        if (first3 === 530 || first3 === 531) return 'VISAKHAPATNAM, ANDHRA PRADESH';
        if (first3 === 532) return 'SRIKAKULAM, ANDHRA PRADESH';
        if (first3 === 533) return 'EAST GODAVARI, ANDHRA PRADESH';
        if (first3 === 534) return 'WEST GODAVARI, ANDHRA PRADESH';
        if (first3 === 535) return 'VIZIANAGARAM, ANDHRA PRADESH';
        if (first3 === 520 || first3 === 521) return 'KRISHNA (VIJAYAWADA), ANDHRA PRADESH';
        if (first3 === 522) return 'GUNTUR, ANDHRA PRADESH';
        if (first3 === 523) return 'PRAKASAM, ANDHRA PRADESH';
        if (first3 === 524) return 'NELLORE, ANDHRA PRADESH';
        if (first3 === 515) return 'ANANTAPUR, ANDHRA PRADESH';
        if (first3 === 516) return 'KADAPA, ANDHRA PRADESH';
        if (first3 === 517) return 'CHITTOOR, ANDHRA PRADESH';
        if (first3 === 518) return 'KURNOOL, ANDHRA PRADESH';
        return 'ANDHRA PRADESH';
    }

    if (first2 >= 50 && first2 <= 50) return 'HYDERABAD, TELANGANA';
    if (first2 >= 11 && first2 <= 11) return 'DELHI, DELHI';
    if (first2 >= 12 && first2 <= 13) return 'HARYANA';
    if (first2 >= 14 && first2 <= 15) return 'PUNJAB';
    if (first2 >= 16 && first2 <= 16) return 'CHANDIGARH';
    if (first2 >= 17 && first2 <= 17) return 'HIMACHAL PRADESH';
    if (first2 >= 18 && first2 <= 19) return 'JAMMU & KASHMIR';
    if (first2 >= 20 && first2 <= 28) return 'UTTAR PRADESH';
    if (first2 >= 30 && first2 <= 34) return 'RAJASTHAN';
    if (first2 >= 36 && first2 <= 39) return 'GUJARAT';
    if (first2 >= 40 && first2 <= 44) return 'MAHARASHTRA';
    if (first2 >= 45 && first2 <= 49) return 'MADHYA PRADESH';
    if (first2 >= 56 && first2 <= 59) return 'KARNATAKA';
    if (first2 >= 60 && first2 <= 64) return 'TAMIL NADU';
    if (first2 >= 67 && first2 <= 69) return 'KERALA';
    if (first2 >= 70 && first2 <= 74) return 'WEST BENGAL';
    if (first2 >= 75 && first2 <= 77) return 'ODISHA';
    if (first2 >= 78 && first2 <= 79) return 'ASSAM';
    if (first2 >= 80 && first2 <= 85) return 'BIHAR';
    if (first2 >= 82 && first2 <= 83) return 'JHARKHAND';
    return 'INDIA';
}

// Mobile validation
$('#mobile').addEventListener('input', (e) => {
    e.target.value = e.target.value.replace(/\D/g, '');
    AppState.mobile = e.target.value;
});

$('#fullName').addEventListener('input', (e) => {
    AppState.fullName = e.target.value;
});

$('#email').addEventListener('input', (e) => {
    AppState.email = e.target.value;
});

$('#annualIncome').addEventListener('input', (e) => {
    e.target.value = e.target.value.replace(/\D/g, '');
    AppState.annualIncome = e.target.value;
});

// Step 2 Next (Location)
$('#step2Next').addEventListener('click', () => {
    if (!AppState.pincode || AppState.pincode.length < 6) {
        pincodeInput.classList.add('invalid');
        pincodeInput.focus();
        showToast('Please enter a valid PIN code');
        return;
    }
    goToStep(3);
});

// Step 3 Next (Identity)
$('#step3Next').addEventListener('click', () => {
    if (!AppState.fullName || AppState.fullName.trim().length < 2) {
        $('#fullName').focus();
        showToast('Please enter your full name');
        return;
    }
    if (!AppState.email || !AppState.email.includes('@')) {
        $('#email').focus();
        showToast('Please enter a valid email address');
        return;
    }
    if (!AppState.annualIncome || parseInt(AppState.annualIncome) <= 0) {
        $('#annualIncome').focus();
        showToast('Please enter your annual income');
        return;
    }
    goToStep(4);
});

// Step 4 Next (Contact & Send OTP)
$('#step4Next').addEventListener('click', async () => {
    if (!AppState.mobile || AppState.mobile.length < 10) {
        $('#mobile').focus();
        showToast('Please enter a valid mobile number');
        return;
    }

    const btn = $('#step4Next');
    const originalContent = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = 'Sending Code... <div class="btn-loader"></div>';

    try {
        const response = await fetch('/api/send-otp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mobile: AppState.mobile })
        });

        const result = await response.json();

        if (result.success) {
            AppState.generatedOtp = result.otp_debug;

            // Toggle sections within Step 4
            $('#mobileSection').style.display = 'none';
            $('#otpSection').style.display = 'block';
            $('#step4Next').style.display = 'none';
            $('#verifyOtpBtn').style.display = 'flex';

            $('#otpPhoneDisplay').textContent = `+91 ${AppState.mobile}`;
            $('#headerSubtitle').textContent = "We've sent a code to your phone. Let's make sure it's you!";
            startOtpTimer();

            setTimeout(() => {
                alert(`🔒 Omegle Health Security\n\nYour 4-digit code is: ${AppState.generatedOtp}`);
            }, 800);
        } else {
            showToast(result.message || 'Error sending code');
        }
    } catch (err) {
        showToast('Connection error. Please try again.');
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalContent;
    }
});

// Back from Step 4
$('#backBtn4').addEventListener('click', () => {
    if ($('#otpSection').style.display === 'block') {
        // Revert to mobile input
        $('#otpSection').style.display = 'none';
        $('#mobileSection').style.display = 'block';
        $('#verifyOtpBtn').style.display = 'none';
        $('#step4Next').style.display = 'flex';
        $('#headerSubtitle').textContent = "Your mobile number is required to send the official quote.";
        clearInterval(otpTimerInterval);
    } else {
        goToStep(3, true);
    }
});

// OTP Logic
const otpFields = $$('.otp-field');

async function verifyOtp() {
    const enteredOtp = Array.from(otpFields).map(f => f.value).join('');

    if (enteredOtp.length < 4) return;

    // Update button state
    const btn = $('#verifyOtpBtn');
    const originalContent = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = 'Verifying... <div class="btn-loader"></div>';

    try {
        const response = await fetch('/api/verify-otp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mobile: AppState.mobile,
                otp: enteredOtp
            })
        });

        const result = await response.json();

        if (result.success) {
            showToast('Verification Successful! ✅');

            setTimeout(() => {
                btn.innerHTML = originalContent;
                btn.disabled = false;
                otpFields.forEach(f => f.value = '');
                goToStep(5);
            }, 500);
        } else {
            showToast('Invalid Security Code! Please try again. ❌');
            btn.innerHTML = originalContent;
            btn.disabled = false;
            // Clear fields so they can retry
            otpFields.forEach(f => f.value = '');
            otpFields[0].focus();
        }
    } catch (err) {
        showToast('Verification failed. Please try again.');
        btn.disabled = false;
        btn.innerHTML = originalContent;
    }
}

otpFields.forEach((field, index) => {
    field.addEventListener('input', (e) => {
        const val = e.target.value;
        // Only allow numbers
        e.target.value = val.replace(/\D/g, '');

        if (e.target.value.length === 1) {
            if (index < otpFields.length - 1) {
                otpFields[index + 1].focus();
            } else {
                // All digits entered - auto verify
                verifyOtp();
            }
        }
    });

    field.addEventListener('keydown', (e) => {
        if (e.key === 'Backspace' && !field.value && index > 0) {
            otpFields[index - 1].focus();
        }
    });
});

let otpTimerInterval;
function startOtpTimer() {
    let timeLeft = 25;
    const btn = $('#resendOtp');
    btn.disabled = true;

    clearInterval(otpTimerInterval);
    otpTimerInterval = setInterval(() => {
        timeLeft--;
        if (timeLeft <= 0) {
            clearInterval(otpTimerInterval);
            btn.innerHTML = 'Resend OTP';
            btn.disabled = false;
        } else {
            btn.innerHTML = `Resend OTP in <span>${timeLeft}s</span>`;
        }
    }, 1000);
}

$('#resendOtp').addEventListener('click', async () => {
    const btn = $('#resendOtp');
    btn.disabled = true;
    btn.innerHTML = 'Sending...';

    try {
        const response = await fetch('/api/send-otp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mobile: AppState.mobile })
        });

        const result = await response.json();

        if (result.success) {
            AppState.generatedOtp = result.otp_debug;
            otpFields.forEach(f => f.value = '');
            otpFields[0].focus();
            showToast('New code sent! 📨');
            startOtpTimer();

            setTimeout(() => {
                alert(`🔒 Omegle Health Security\n\nYour NEW code is: ${AppState.generatedOtp}`);
            }, 500);
        }
    } catch (err) {
        showToast('Failed to resend. Check connection.');
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Resend OTP';
    }
});



$('#verifyOtpBtn').addEventListener('click', () => {
    const otp = Array.from(otpFields).map(f => f.value).join('');
    if (otp.length < 4) {
        showToast('Please enter the 4-digit code');
        return;
    }
    verifyOtp();
});

// ============================================
// Step 3: Member Selection
// ============================================
// Gender toggle
$$('.gender-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        $$('.gender-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        AppState.gender = btn.dataset.gender;

        // Update spouse label
        const wifeLabel = $('#memberWife').nextElementSibling;
        if (AppState.gender === 'female') {
            wifeLabel.textContent = 'Husband';
        } else {
            wifeLabel.textContent = 'Wife';
        }
    });
});

// Member card selection
$$('.member-card').forEach(card => {
    card.addEventListener('click', (e) => {
        // Don't toggle if clicking count buttons
        if (e.target.closest('.count-btn')) return;

        const member = card.dataset.member;
        card.classList.toggle('selected');

        if (card.classList.contains('selected')) {
            if (!AppState.selectedMembers.includes(member)) {
                AppState.selectedMembers.push(member);
            }
            // Show count for son/daughter
            if (member === 'son') {
                $('#sonCount').style.display = 'flex';
            }
            if (member === 'daughter') {
                $('#daughterCount').style.display = 'flex';
            }
        } else {
            AppState.selectedMembers = AppState.selectedMembers.filter(m => m !== member);
            if (member === 'son') {
                $('#sonCount').style.display = 'none';
                AppState.sonCount = 1;
                $('#sonValue').textContent = '1';
            }
            if (member === 'daughter') {
                $('#daughterCount').style.display = 'none';
                AppState.daughterCount = 1;
                $('#daughterValue').textContent = '1';
            }
        }
    });
});

// Son/Daughter count
$('#sonPlus').addEventListener('click', (e) => {
    e.stopPropagation();
    if (AppState.sonCount < 4) {
        AppState.sonCount++;
        $('#sonValue').textContent = AppState.sonCount;
    }
});

$('#sonMinus').addEventListener('click', (e) => {
    e.stopPropagation();
    if (AppState.sonCount > 1) {
        AppState.sonCount--;
        $('#sonValue').textContent = AppState.sonCount;
    }
});

$('#daughterPlus').addEventListener('click', (e) => {
    e.stopPropagation();
    if (AppState.daughterCount < 4) {
        AppState.daughterCount++;
        $('#daughterValue').textContent = AppState.daughterCount;
    }
});

$('#daughterMinus').addEventListener('click', (e) => {
    e.stopPropagation();
    if (AppState.daughterCount > 1) {
        AppState.daughterCount--;
        $('#daughterValue').textContent = AppState.daughterCount;
    }
});

// View more members
$('#viewMoreBtn').addEventListener('click', () => {
    const extra = $('#extraMembers');
    const btn = $('#viewMoreBtn');
    if (extra.style.display === 'none') {
        extra.style.display = 'block';
        btn.innerHTML = 'View less members <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="18 15 12 9 6 15"/></svg>';
    } else {
        extra.style.display = 'none';
        btn.innerHTML = 'View more members <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="6 9 12 15 18 9"/></svg>';
    }
});

// Step 5 Next (Members)
$('#step5Next').addEventListener('click', () => {
    if (AppState.selectedMembers.length === 0) {
        showToast('Please select at least one family member');
        return;
    }
    buildAgeEntries();
    goToStep(6);
});

// ============================================
// Step 4: Age Details
// ============================================
function buildAgeEntries() {
    const container = $('#ageEntries');
    container.innerHTML = '';
    AppState.memberDetails = [];

    const memberLabels = {
        self: 'Self',
        wife: AppState.gender === 'female' ? 'Husband' : 'Wife',
        husband: 'Husband',
        son: 'Son',
        daughter: 'Daughter',
        father: 'Father',
        mother: 'Mother',
        fatherInLaw: 'Father-in-law',
        motherInLaw: 'Mother-in-law',
    };

    const memberIcons = {
        self: '',
        wife: '',
        husband: '',
        son: '',
        daughter: '',
        father: '',
        mother: '',
        fatherInLaw: '',
        motherInLaw: '',
    };

    // Build member list with counts
    const memberList = [];
    for (const member of AppState.selectedMembers) {
        if (member === 'son') {
            for (let i = 0; i < AppState.sonCount; i++) {
                memberList.push({
                    type: 'son',
                    label: AppState.sonCount > 1 ? `Son ${i + 1}` : 'Son',
                    icon: memberIcons.son,
                });
            }
        } else if (member === 'daughter') {
            for (let i = 0; i < AppState.daughterCount; i++) {
                memberList.push({
                    type: 'daughter',
                    label: AppState.daughterCount > 1 ? `Daughter ${i + 1}` : 'Daughter',
                    icon: memberIcons.daughter,
                });
            }
        } else {
            memberList.push({
                type: member,
                label: memberLabels[member] || member,
                icon: memberIcons[member] || memberIcons.self,
            });
        }
    }

    memberList.forEach((member, index) => {
        const entry = document.createElement('div');
        entry.classList.add('age-entry');
        entry.dataset.index = index;

        entry.innerHTML = `
            <div class="age-avatar">
                <span class="age-avatar-badge">${member.label.charAt(0)}</span>
                <span class="age-avatar-label">${member.label}</span>
            </div>
            <div class="age-details">
                <input type="date" class="age-dob-input" data-index="${index}" id="dob_${index}" placeholder="DD/MM/YYYY">
                <div class="age-display" id="ageDisplay_${index}"></div>
            </div>
        `;

        container.appendChild(entry);

        AppState.memberDetails.push({
            type: member.type,
            label: member.label,
            dob: '',
            age: 0,
        });

        // DOB event
        const dobInput = entry.querySelector('.age-dob-input');
        dobInput.addEventListener('change', (e) => {
            const dob = new Date(e.target.value);
            const today = new Date();
            let age = today.getFullYear() - dob.getFullYear();
            const m = today.getMonth() - dob.getMonth();
            if (m < 0 || (m === 0 && today.getDate() < dob.getDate())) {
                age--;
            }
            if (age >= 0 && age < 150) {
                AppState.memberDetails[index].dob = e.target.value;
                AppState.memberDetails[index].age = age;
                $(`#ageDisplay_${index}`).textContent = `Age: ${age} Years`;
            }
        });
    });
}

// Add Member button in Step 6
$('#addMemberBtn').addEventListener('click', () => {
    goToStep(5, true);
});

// Health conditions
$$('.condition-chip input').forEach(input => {
    input.addEventListener('change', () => {
        if (input.value === 'none' && input.checked) {
            // Uncheck all others
            $$('.condition-chip input').forEach(i => {
                if (i.value !== 'none') i.checked = false;
            });
            AppState.conditions = ['none'];
        } else {
            // Uncheck "none"
            $('#condNoneInput').checked = false;
            AppState.conditions = [];
            $$('.condition-chip input:checked').forEach(i => {
                AppState.conditions.push(i.value);
            });
        }
    });
});

// Smoker toggle
$('#smokerInput').addEventListener('change', (e) => {
    AppState.isSmoker = e.target.checked;
});

// BMI
$('#bmiValue').addEventListener('input', (e) => {
    AppState.bmi = parseFloat(e.target.value) || null;
});

// ============================================
// API Request Builder
// ============================================
function buildApiPayload() {
    return {
        fullName: AppState.fullName,
        mobile: AppState.mobile,
        email: AppState.email,
        members: AppState.memberDetails.map(m => ({
            type: m.type,
            age: m.age,
            label: m.label,
        })),
        conditions: AppState.conditions.filter(c => c !== 'none'),
        smoker: AppState.isSmoker,
        bmi: AppState.bmi || 24.0,
        annualIncome: AppState.annualIncome,
        sum_insured: AppState.sumInsured,
        tenure: AppState.tenure,
        pincode: AppState.pincode,
        gender: AppState.gender,
        num_children: AppState.memberDetails.filter(m => m.type === 'son' || m.type === 'daughter').length,
    };
}

// ============================================
// View Plans - Prediction (calls ML backend)
// ============================================
$('#viewPlansBtn').addEventListener('click', async () => {
    // Validate ages
    let allAgesSet = true;
    for (const member of AppState.memberDetails) {
        if (!member.dob || member.age <= 0) {
            allAgesSet = false;
            break;
        }
    }

    if (!allAgesSet) {
        showToast('Please enter date of birth for all members');
        return;
    }

    // Show loading
    showLoading();

    try {
        // Try ML backend first
        if (backendAvailable) {
            const payload = buildApiPayload();
            const response = await fetch(`${API_BASE_URL}/api/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            const result = await response.json();

            if (result.success) {
                hideLoading();
                lastApiPlans = result.plans;
                generateResults(result.plans, result.prediction_details);
                goToStep(7);
                createConfetti();
                console.log('✅ ML Prediction successful', result.prediction_details);
                return;
            } else {
                console.warn('Backend error:', result.error);
            }
        }

        // Fallback: client-side prediction
        console.log('Using client-side fallback prediction');
        setTimeout(() => {
            hideLoading();
            const plans = predictor.predict(AppState);
            // Convert to API format for consistency
            const apiFormatPlans = plans.map(p => ({
                ...p,
                original_price: p.originalPrice,
                gst_amount: Math.round(p.premium * 0.18),
                total_with_gst: Math.round(p.premium * 1.18),
            }));
            generateResults(apiFormatPlans, null);
            goToStep(7);
            createConfetti();
        }, 1500);

    } catch (error) {
        console.error('Prediction error:', error);
        // Fallback to client-side
        hideLoading();
        const plans = predictor.predict(AppState);
        const apiFormatPlans = plans.map(p => ({
            ...p,
            original_price: p.originalPrice,
            gst_amount: Math.round(p.premium * 0.18),
            total_with_gst: Math.round(p.premium * 1.18),
        }));
        generateResults(apiFormatPlans, null);
        goToStep(7);
        createConfetti();
    }
});

function showLoading() {
    const overlay = document.createElement('div');
    overlay.classList.add('loading-overlay');
    overlay.id = 'loadingOverlay';
    overlay.innerHTML = `
        <div class="loading-spinner"></div>
        <div class="loading-text">${backendAvailable ? '🧠 AI Model analyzing your profile...' : 'Analyzing your profile...'}</div>
        <div class="loading-subtext">${backendAvailable ? 'XGBoost ML model processing 25 health factors' : 'Comparing 100+ plans for best rates'}</div>
    `;
    document.body.appendChild(overlay);
}

function hideLoading() {
    const overlay = $('#loadingOverlay');
    if (overlay) overlay.remove();
}

// ============================================
// Results Generation
// ============================================
function generateResults(plans, predictionDetails) {
    // Update policy members display
    const memberNames = AppState.memberDetails.map(m => m.label);
    let membersText = memberNames.slice(0, 2).join(', ');
    if (memberNames.length > 2) {
        membersText += ` +${memberNames.length - 2}`;
    }
    if (AppState.pincode) {
        membersText += ` — ${AppState.pincode}`;
    }
    $('#policyMembers').textContent = membersText;

    // Show ML badge if backend was used
    if (predictionDetails) {
        showMlBadge(predictionDetails);
    }

    // Render plan cards
    renderPlanCards(plans);

    // Render features
    renderFeatures(plans);

    // Render risk factors if available
    if (predictionDetails && predictionDetails.risk_factors) {
        renderRiskFactors(predictionDetails.risk_factors);
    }
}

function showMlBadge(details) {
    const existing = document.getElementById('mlBadge');
    if (existing) existing.remove();

    const badge = document.createElement('div');
    badge.id = 'mlBadge';
    badge.className = 'ml-badge';
    badge.innerHTML = `
        <span class="ml-badge-icon">🧠</span>
        <span class="ml-badge-text"><strong class="ml-badge-title">AI-Powered Prediction</strong> — XGBoost ML model analyzed ${details.total_members} member(s), ${details.conditions_count} condition(s), BMI ${details.bmi}</span>
    `;
    const policyInfo = document.getElementById('policyInfo');
    if (policyInfo) policyInfo.after(badge);
}

function renderRiskFactors(factors) {
    const existing = document.getElementById('riskFactors');
    if (existing) existing.remove();

    if (!factors || factors.length === 0) return;

    const container = document.createElement('div');
    container.id = 'riskFactors';
    container.className = 'risk-factors-container';

    const impactColors = {
        very_high: '#dc2626',
        high: '#ef4444',
        medium: '#f97316',
        low: '#eab308',
        positive: '#14b8a6',
    };

    container.innerHTML = `
        <h4 class="risk-analysis-title">📋 Risk Analysis</h4>
        <div class="risk-factors-list">
            ${factors.map(f => `
                <div class="risk-factor-item">
                    <span class="risk-dot" style="background:${impactColors[f.impact] || '#94a3b8'};"></span>
                    <strong class="risk-name">${f.factor}</strong>
                    <span class="risk-desc">— ${f.description}</span>
                </div>
            `).join('')}
        </div>
    `;

    const featuresTable = document.getElementById('featuresTable');
    if (featuresTable) featuresTable.before(container);
}

function renderPlanCards(plans) {
    const container = $('#plansContainer');
    container.innerHTML = '';

    // Show top 2 plans (Premier and Premier Plus)
    const displayPlans = plans.slice(1); // Skip essential, show Premier and Premier Plus

    displayPlans.forEach((plan, index) => {
        const card = document.createElement('div');
        card.classList.add('plan-card');
        if (index === 0) {
            card.classList.add('selected', 'recommended');
            AppState.selectedPlan = index;
        }
        card.dataset.planIndex = index;

        // Support both API format (original_price) and client format (originalPrice)
        const originalPrice = plan.original_price || plan.originalPrice || 0;
        const gstAmount = plan.gst_amount || Math.round(plan.premium * 0.18);

        card.innerHTML = `
            ${plan.recommended ? '<div class="recommended-badge">Recommended</div>' : ''}
            <div class="plan-radio"></div>
            <h3 class="plan-name">${plan.name}</h3>
            <div class="plan-discount">${plan.discount}% Off</div>
            <div class="plan-original-price">₹${originalPrice.toLocaleString('en-IN')}</div>
            <div class="plan-price"><span class="currency">₹</span>${plan.premium.toLocaleString('en-IN')}</div>
            <div class="plan-gst">+ GST ₹${gstAmount.toLocaleString('en-IN')}</div>
        `;

        card.addEventListener('click', () => {
            $$('.plan-card').forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
            AppState.selectedPlan = index;
        });

        container.appendChild(card);
    });
}

function renderFeatures(plans) {
    const container = $('#featuresTable');
    container.innerHTML = '';

    const displayPlans = plans.slice(1);

    // Feature rows
    const features = [
        { label: 'Plan Type', values: displayPlans.map(p => p.description) },
        { label: 'Health Checkup', values: ['Health Checkup', 'Health Checkup'] },
        { label: 'Consumables', values: ['Consumables covered', 'Consumables covered'] },
        { label: 'Hospital Bills', values: ['100% hospital bills paid* - No co-payment or room rent capping', '100% hospital bills paid* - No co-payment or room rent capping'] },
    ];

    features.forEach(feature => {
        const row = document.createElement('div');
        row.classList.add('feature-row');
        row.innerHTML = feature.values.map((v, i) => `
            <div class="feature-cell ${i === 0 ? 'highlight' : ''}">${v}</div>
        `).join('');
        container.appendChild(row);
    });

    // View benefits buttons
    const btnRow = document.createElement('div');
    btnRow.classList.add('feature-row');
    btnRow.innerHTML = displayPlans.map((plan, i) => `
        <div class="feature-cell">
            <button class="view-benefits-btn" data-plan="${i}">View all benefits →</button>
        </div>
    `).join('');
    container.appendChild(btnRow);

    // Bind benefit buttons
    btnRow.querySelectorAll('.view-benefits-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            showBenefitsModal(displayPlans[parseInt(btn.dataset.plan)]);
        });
    });
}

// ============================================
// Benefits Modal
// ============================================
function showBenefitsModal(plan) {
    const benefits = [
        { icon: '🏥', title: 'Hospitalization Cover', desc: `Up to ₹${AppState.sumInsured}L coverage for hospitalization expenses` },
        { icon: '🔬', title: 'Pre & Post Hospitalization', desc: '60 days pre and 180 days post hospitalization expenses covered' },
        { icon: '💊', title: 'Consumables Covered', desc: 'Gloves, PPE kits, syringes and other consumables covered' },
        { icon: '🏠', title: 'No Room Rent Capping', desc: 'Choose any room category without any capping on rent' },
        { icon: '🚑', title: 'Ambulance Charges', desc: 'Up to ₹5,000 per hospitalization for ambulance expenses' },
        { icon: '🔄', title: 'Restoration Benefit', desc: '100% restoration of sum insured once in a policy year' },
        { icon: '🌿', title: 'AYUSH Treatment', desc: 'Ayurveda, Yoga, Unani, Siddha, Homeopathy treatments covered' },
        { icon: '👶', title: 'Maternity Cover', desc: 'Available as an add-on for delivery expenses' },
        { icon: '🏋️', title: 'Annual Health Checkup', desc: 'Free annual health checkup for all insured members' },
        { icon: '📋', title: 'Day Care Procedures', desc: '500+ day care procedures covered without 24hr hospitalization' },
    ];

    const overlay = document.createElement('div');
    overlay.classList.add('modal-overlay');
    overlay.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h2 class="modal-title">${plan.name}</h2>
                <button class="modal-close" id="modalClose">✕</button>
            </div>
            ${benefits.map(b => `
                <div class="benefit-item">
                    <div class="benefit-icon">${b.icon}</div>
                    <div class="benefit-text">
                        <h4>${b.title}</h4>
                        <p>${b.desc}</p>
                    </div>
                </div>
            `).join('')}
        </div>
    `;

    document.body.appendChild(overlay);

    overlay.querySelector('#modalClose').addEventListener('click', () => overlay.remove());
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) overlay.remove();
    });
}

// ============================================
// Confetti Effect
// ============================================
function createConfetti() {
    const container = $('#confettiContainer');
    if (!container) return;
    container.innerHTML = '';

    const colors = ['#0E1B48', '#C18DB4', '#E2CAD8', '#87A7D0', '#27425D', '#ff6b6b', '#4ecdc4'];
    const shapes = ['circle', 'rect'];

    for (let i = 0; i < 30; i++) {
        const piece = document.createElement('div');
        piece.classList.add('confetti-piece');
        const color = colors[Math.floor(Math.random() * colors.length)];
        const shape = shapes[Math.floor(Math.random() * shapes.length)];
        const size = Math.random() * 8 + 4;

        piece.style.cssText = `
            left: ${Math.random() * 100}%;
            top: ${Math.random() * 40}%;
            width: ${size}px;
            height: ${shape === 'circle' ? size : size * 0.6}px;
            background: ${color};
            border-radius: ${shape === 'circle' ? '50%' : '2px'};
            animation-delay: ${Math.random() * 0.5}s;
            animation-duration: ${Math.random() * 1.5 + 1.5}s;
        `;
        container.appendChild(piece);
    }
}

// ============================================
// Plan Controls
// ============================================
// Tenure buttons
$$('.tenure-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        $$('.tenure-btn').forEach(b => {
            b.classList.remove('active');
            b.innerHTML = b.dataset.tenure + (btn.dataset.tenure === '1' ? ' yr' : ' yrs');
        });
        btn.classList.add('active');
        btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg> ${btn.dataset.tenure} ${btn.dataset.tenure === '1' ? 'yr' : 'yrs'}`;

        AppState.tenure = parseInt(btn.dataset.tenure);
        recalculatePlans();
    });
});

// Find right value button
$('#findValueBtn').addEventListener('click', () => {
    const slider = $('#sumSliderContainer');
    slider.style.display = slider.style.display === 'none' ? 'block' : 'none';
});

// Sum slider
$('#sumSlider').addEventListener('input', (e) => {
    AppState.sumInsured = parseInt(e.target.value);
    $('#sumAmount').textContent = AppState.sumInsured;
    recalculatePlans();
});

async function recalculatePlans() {
    try {
        if (backendAvailable) {
            const payload = buildApiPayload();
            const response = await fetch(`${API_BASE_URL}/api/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const result = await response.json();
            if (result.success) {
                lastApiPlans = result.plans;
                renderPlanCards(result.plans);
                renderFeatures(result.plans);
                return;
            }
        }
    } catch (e) {
        console.warn('Recalculation fallback to client-side');
    }
    // Fallback
    const plans = predictor.predict(AppState);
    const apiFormatPlans = plans.map(p => ({
        ...p,
        original_price: p.originalPrice,
        gst_amount: Math.round(p.premium * 0.18),
        total_with_gst: Math.round(p.premium * 1.18),
    }));
    renderPlanCards(apiFormatPlans);
    renderFeatures(apiFormatPlans);
}

// ============================================
// Back Navigation
// ============================================
$('#backBtn1').addEventListener('click', () => goToStep(0, true));
$('#backBtn2').addEventListener('click', () => goToStep(1, true));
$('#backBtn3').addEventListener('click', () => goToStep(2, true));
$('#backBtn4').addEventListener('click', () => goToStep(3, true));
$('#backBtn5').addEventListener('click', () => goToStep(4, true));
$('#backBtn6').addEventListener('click', () => goToStep(5, true));

// Banner close
$('#bannerClose').addEventListener('click', () => {
    $('#successBanner').style.display = 'none';
});

// Edit policy
$('#editPolicyBtn').addEventListener('click', () => {
    goToStep(5, true);
});

// Start over
$('#startOverBtn').addEventListener('click', () => {
    location.reload();
});

// ============================================
// Download Quote
// ============================================
$('#downloadQuoteBtn').addEventListener('click', () => {
    generatePDFQuote();
});

function generatePDFQuote() {
    const plans = predictor.predict(AppState);
    const selectedPlan = plans.slice(1)[AppState.selectedPlan] || plans[1];

    const quoteData = {
        name: AppState.fullName || 'Customer',
        members: AppState.memberDetails.map(m => `${m.label} (Age: ${m.age})`).join(', '),
        plan: selectedPlan.name,
        premium: `₹${selectedPlan.premium.toLocaleString('en-IN')}`,
        sumInsured: `₹${AppState.sumInsured} Lacs`,
        tenure: `${AppState.tenure} Year(s)`,
        location: AppState.location,
        date: new Date().toLocaleDateString('en-IN'),
    };

    // Create downloadable text quote
    const quoteText = `
╔══════════════════════════════════════════════════╗
║       MEDISHIELD PRO - INSURANCE QUOTE           ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  Quote Date: ${quoteData.date.padEnd(35)}║
║  Quote #: MSP-${Date.now().toString().slice(-8).padEnd(31)}║
║                                                  ║
╠══════════════════════════════════════════════════╣
║  CUSTOMER DETAILS                                ║
╠══════════════════════════════════════════════════╣
║  Name: ${quoteData.name.padEnd(41)}║
║  Location: ${quoteData.location.padEnd(37)}║
║  Members: ${quoteData.members.substring(0, 37).padEnd(37)}║
║                                                  ║
╠══════════════════════════════════════════════════╣
║  PLAN DETAILS                                    ║
╠══════════════════════════════════════════════════╣
║  Selected Plan: ${quoteData.plan.padEnd(31)}║
║  Sum Insured: ${quoteData.sumInsured.padEnd(33)}║
║  Policy Tenure: ${quoteData.tenure.padEnd(31)}║
║  Annual Premium: ${quoteData.premium.padEnd(30)}║
║  GST (18%): Additional                          ║
║                                                  ║
╠══════════════════════════════════════════════════╣
║  * This is an indicative quote. Final premium    ║
║    may vary based on medical underwriting.       ║
║  * Terms & Conditions apply.                     ║
╚══════════════════════════════════════════════════╝
    `.trim();

    // Download as text file
    const blob = new Blob([quoteText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `MediShield_Quote_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);

    showToast('Quote downloaded successfully!');
}

// ============================================
// Toast Notification
// ============================================
function showToast(message) {
    const toast = $('#toast');
    $('#toastMessage').textContent = message;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 3000);
}

// ============================================
// Initialize
// ============================================
// ============================================
// Initialize
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    createParticles();
    initTheme();
    initBmiCalculator();
    goToStep(0);
});

// Theme Logic
function initTheme() {
    const themeToggle = document.getElementById('themeToggle');
    const savedTheme = localStorage.getItem('theme') || 'light';

    document.documentElement.setAttribute('data-theme', savedTheme);

    themeToggle.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';

        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);

        // Add a nice ripple or sparkle effect? Let's just toast for now
        showToast(`Switched to ${newTheme} mode`);
    });
}

// BMI Tool Logic
function initBmiCalculator() {
    const heightInput = document.getElementById('bmiHeight');
    const weightInput = document.getElementById('bmiWeight');
    const resultBadge = document.getElementById('bmiResult');
    const hiddenBmi = document.getElementById('bmiValue');

    function calculate() {
        const h = parseFloat(heightInput.value) / 100;
        const w = parseFloat(weightInput.value);

        if (h > 0 && w > 0) {
            const bmi = (w / (h * h)).toFixed(1);
            resultBadge.textContent = bmi;
            hiddenBmi.value = bmi;
            AppState.bmi = parseFloat(bmi);

            // Color feedback
            if (bmi < 18.5) resultBadge.style.background = '#87A7D0'; // Underweight (Soft Blue)
            else if (bmi < 25) resultBadge.style.background = '#45b7af'; // Normal (Teal)
            else if (bmi < 30) resultBadge.style.background = '#C18DB4'; // Overweight (Pink)
            else resultBadge.style.background = '#27425D'; // Obese (Slate)
        }
    }

    heightInput.addEventListener('input', calculate);
    weightInput.addEventListener('input', calculate);
}
