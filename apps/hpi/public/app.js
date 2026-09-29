const app = {
    currentFile: null,

    init() {
        // Setup Drag & Drop
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('file-input');

        dropzone.addEventListener('click', () => fileInput.click());
        
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.style.borderColor = 'var(--primary)';
        });
        
        dropzone.addEventListener('dragleave', () => {
            dropzone.style.borderColor = 'rgba(204, 240, 80, 0.3)';
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.style.borderColor = 'rgba(204, 240, 80, 0.3)';
            if (e.dataTransfer.files.length) {
                this.handleFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                this.handleFile(e.target.files[0]);
            }
        });
    },

    showScreen(id) {
        document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
        document.getElementById(id).classList.remove('hidden');
        // small timeout to allow display:block to apply before animating opacity
        setTimeout(() => {
            document.getElementById(id).classList.add('active');
        }, 10);
    },

    showToast(msg) {
        const toast = document.getElementById('toast');
        toast.textContent = msg;
        toast.classList.add('show');
        setTimeout(() => toast.classList.remove('show'), 3000);
    },

    async handleFile(file) {
        if (!file.type.startsWith('image/')) {
            this.showToast('Please upload an image file');
            return;
        }
        
        this.currentFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('preview-img').src = e.target.result;
        };
        reader.readAsDataURL(file);

        // Upload to Analyze endpoint
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            // Set loading states
            document.getElementById('quality-val').textContent = 'Checking...';
            document.getElementById('confidence-val').textContent = '...';
            
            this.showScreen('screen-review');
            
            const res = await fetch('/notes/analyze/', {
                method: 'POST',
                body: formData
            });
            
            if (!res.ok) throw new Error('Analysis failed');
            
            const data = await res.json();
            
            const qVal = document.getElementById('quality-val');
            qVal.textContent = data.scan_quality.charAt(0).toUpperCase() + data.scan_quality.slice(1);
            qVal.className = `value ${data.scan_quality}`;
            
            document.getElementById('confidence-val').textContent = `${data.handwriting_detected}%`;
            
            const btnProceed = document.getElementById('btn-proceed');
            const warning = document.getElementById('retake-warning');
            
            if (data.retake) {
                btnProceed.disabled = true;
                warning.classList.remove('hidden');
            } else {
                btnProceed.disabled = false;
                warning.classList.add('hidden');
            }
            
        } catch (err) {
            console.error(err);
            this.showToast('Failed to analyze image. Please try again.');
            this.showScreen('screen-home');
        }
    },

    startAnalysis() {
        this.showScreen('screen-analysis');
        document.getElementById('stream-steps').innerHTML = '';
        
        const formData = new FormData();
        formData.append('file', this.currentFile);
        
        // Use fetch to read stream
        fetch('/notes/upload_stream/', {
            method: 'POST',
            headers: {
                'x-device-id': 'web-demo'
            },
            body: formData
        }).then(async response => {
            if (!response.ok) {
                const detail = await response.text().catch(() => '');
                throw new Error(`HTTP ${response.status}${detail ? ': ' + detail.slice(0, 120) : ''}`);
            }
            if (!response.body) {
                throw new Error('No response body (stream unsupported)');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            let buffer = '';
            let stepCounter = 1;
            let sawEvent = false;
            
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                
                // Keep the last partial chunk in the buffer
                buffer = lines.pop() || '';
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.substring(6);
                        try {
                            const data = JSON.parse(dataStr);
                            sawEvent = true;
                            if (data.status === 'Failed') {
                                throw new Error(data.error || data.message || 'Analysis failed');
                            }
                            if (data.status === 'Complete') {
                                this.renderProfile(data.result.profile);
                            } else {
                                this.addStreamStep(stepCounter++, data.status);
                            }
                        } catch (e) {
                            if (e instanceof SyntaxError) {
                                console.error('JSON parse error', e);
                            } else {
                                throw e;
                            }
                        }
                    }
                }
            }
            if (!sawEvent) {
                throw new Error('Stream closed with no events (stale server on port 8003?)');
            }
        }).catch(err => {
            console.error(err);
            this.showToast(err.message || 'Streaming failed. Check connection.');
        });
    },
    
    addStreamStep(num, text) {
        const container = document.getElementById('stream-steps');
        
        // Mark previous steps as done
        const circles = container.querySelectorAll('.step-circle');
        circles.forEach(c => c.innerHTML = '✓');
        circles.forEach(c => c.classList.add('done'));
        
        const stepHTML = `
            <div class="stream-step">
                <div class="step-circle">${num}</div>
                <div class="step-text">${text}</div>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', stepHTML);
    },
    
    renderProfile(profile) {
        // Wait a sec for the last step animation, then show profile
        setTimeout(() => {
            this.currentProfile = profile;
            document.getElementById('res-standout').textContent = profile.standout_strength;
            
            const scoresContainer = document.getElementById('dynamic-scores');
            scoresContainer.innerHTML = '';
            
            profile.scores.forEach((score, index) => {
                const row = document.createElement('div');
                row.className = 'score-row';
                row.innerHTML = `
                    <div class="score-header">
                        <span>${score.name}</span>
                        <span id="score-val-${index}">${score.value}%</span>
                    </div>
                    <div class="progress-bar"><div class="fill" id="score-fill-${index}"></div></div>
                `;
                scoresContainer.appendChild(row);
            });
            
            this.showScreen('screen-profile');
            
            // Animate bars after transition
            setTimeout(() => {
                profile.scores.forEach((score, index) => {
                    document.getElementById(`score-fill-${index}`).style.width = `${score.value}%`;
                });
            }, 400);
            
            // Pre-load insights screen with the first trait
            this.loadInsight(profile.insights[0].title);
            
        }, 1500);
    },
    
    loadInsight(traitTitle) {
        if (!this.currentProfile) return;
        
        const insight = this.currentProfile.insights.find(i => i.title === traitTitle);
        if (!insight) return;
        
        // Update Insight Card Text
        document.getElementById('insight-title').textContent = insight.title;
        document.getElementById('insight-overall').textContent = insight.overall_interpretation;
        document.getElementById('insight-best').textContent = insight.at_your_best;
        document.getElementById('insight-watch').textContent = insight.what_to_watch;
        document.getElementById('insight-action').textContent = insight.action_to_practise;
        
        // Update Pills
        const pillsContainer = document.getElementById('trait-pills');
        pillsContainer.innerHTML = '';
        
        this.currentProfile.insights.forEach(i => {
            const pill = document.createElement('div');
            pill.className = `trait-pill ${i.title === traitTitle ? 'active' : ''}`;
            pill.textContent = i.title;
            pill.onclick = () => this.loadInsight(i.title);
            pillsContainer.appendChild(pill);
        });
    },

    async startAdvancedAnalysis() {
        if (!this.currentFile) {
            this.showToast('No image selected.');
            return;
        }

        this.showScreen('screen-advanced');

        const loading = document.getElementById('advanced-loading');
        const results = document.getElementById('advanced-results');
        loading.classList.remove('hidden');
        results.classList.add('hidden');
        results.innerHTML = '';

        try {
            const formData = new FormData();
            formData.append('file', this.currentFile);

            const res = await fetch('/notes/advanced_analysis/', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) throw new Error('Advanced analysis failed');

            const data = await res.json();
            loading.classList.add('hidden');

            const characteristics = [
                { icon: '📐', label: 'Slant', key: 'slant' },
                { icon: '↔️', label: 'Spacing', key: 'spacing' },
                { icon: '🔵', label: 'Shapes', key: 'shapes' },
                { icon: '📏', label: 'Relative Size', key: 'relative_size' },
                { icon: '📊', label: 'Consistency', key: 'consistency' },
                { icon: '✒️', label: 'Stroke Geometry', key: 'stroke_geometry' },
            ];

            results.innerHTML = characteristics.map((c, i) => `
                <div class="adv-card" style="animation-delay: ${i * 0.1}s">
                    <div class="adv-icon">${c.icon}</div>
                    <div class="adv-content">
                        <span class="adv-label">${c.label}</span>
                        <span class="adv-value">${data[c.key]}</span>
                    </div>
                </div>
            `).join('');

            results.classList.remove('hidden');

        } catch (err) {
            console.error(err);
            loading.classList.add('hidden');
            this.showToast('Advanced analysis failed. Please try again.');
            this.showScreen('screen-review');
        }
    }
};

document.addEventListener('DOMContentLoaded', () => app.init());
