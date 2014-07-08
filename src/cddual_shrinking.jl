function cddual_shrinking(X::Matrix,
	            Y::Vector;
	            C::Real = 1.0,
	            norm::Integer = 2,
	            randomized::Bool = true,
	            maxpasses::Integer = 2,
	            error::Real = 0.001)
	# l: # of samples
	# n: # of features
	n, l = size(X)
	alpha = zeros(l)
	w = zeros(n)

	# Set U and D
	#  * L1-SVM: U = C, D[i] = 0
	#  * L2-SVM: U = Inf, D[i] = 1 / (2C)
	U = 0.0
	D = Array(Float64, l)
	if norm == 1
		U = C
		for i in 1:l
			D[i] = 0.0
		end
	elseif norm	== 2
		U = Inf
		for i in 1:l
			D[i] = 1.0 / (2.0 * C)
		end
	else
		DomainError("Only L1-SVM and L2-SVM are supported")
	end

	# Set Qbar
	Qbar = Array(Float64, l)
	for i in 1:l
		Qbar[i] = D[i] + dot(X[:, i], X[:, i])
	end

	# Loop over examples
	converged = false
	pass = 0

	# set M_bar and m_bar
	M_bar = Inf
	m_bar = -Inf
	A = 1:l

	while !converged
		# Assess convergence
		pass += 1
		if pass == maxpasses
			converged = true
		end

		# Choose order of observations to process
		if randomized
			A = randperm(l)
		else
			A = 1:l
		end

		M = -Inf
		m = Inf

		# Process all observations
		for i in A
			g = Y[i] * dot(w, X[:, i]) - 1.0 + D[i] * alpha[i]
			pg = 0

			if alpha[i] <= 0.0
				if g > M_bar
					pop!(A, i)
				if g < 0.0
					pg = g
			elseif alpha[i] >= U
				if g < m_bar
					pop!(A, i)
				if g > 0.0
					pg = g
			else
				pg = g
			end

			M = max(M, pg)
			m = min(m, pg)

			if abs(pg) > 0.0
				alphabar = alpha[i]
				alpha[i] = min(max(alpha[i] - g / Qbar[i], 0.0), U)
				for j in 1:n
					w[j] = w[j] + (alpha[i] - alphabar) * Y[i] * X[j, i]
				end
			end
		end

		if abs(M-m) < error
			if A == 1:l
				break
			else
				A = 1:l
				M_bar = Inf
				m_bar = -Inf
			end
		end

		if M <= 0
			M_bar = Inf
		else
			M_bar = M
		end

		if m >= 0
			m_bar = - Inf
		else
			m_bar = m
		end
	end
	return SVMFit(w, pass, converged)
end
