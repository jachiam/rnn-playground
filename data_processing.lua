
-- DATA MANIPULATION FUNCTIONS
function data(filename)
	local f = torch.DiskFile(filename)
	local rawdata = f:readString('*a')
	f:close()
	return rawdata
end

function split_string(rawdata,pattern,lower,wordopt)
	local replace 
	if wordopt then
		replace = wordopt.replace or false
	end
	if pattern == 'word' then
		--ptrn = '%w+\'?%w*[%s%p]'
		ptrn = '[^%s]+\n?'
	else
		ptrn = pattern
	end
	local breakapart = rawdata:gmatch(ptrn)
	local splitstring = {}
	local tokens = {}
	for elem in breakapart do
		tokens = {}
		if lower then elem = elem:lower() end
		if pattern == 'word' then
			local pref = {}
			local front = elem
			local back = {}

			--[[ strip off newline characters
			back[1] = elem:sub(elem:len(),elem:len())
			if back[1] == '\n' then 
				front = elem:sub(1,elem:len()-1)
			else
				front = elem
			end]]

			-- strip of punctuation characters and newlines
			for i=1,front:len() do
				local prevchar = front:sub(1,1)
				if prevchar:match('[%p\n]') then
					table.insert(pref,prevchar)
					front = front:sub(2,front:len())
				else
					break
				end
			end
			for i=front:len(),1,-1 do
				local lastchar = front:sub(front:len(),front:len())
				if lastchar:match('[%p\n]') then
					table.insert(back,lastchar)
					front = front:sub(1,front:len()-1)
				else
					break
				end
			end

			-- prefix characters/punctuation to tokens
			for i=1,#pref do
				tokens[#tokens+1] = pref[i]
			end

			-- word to token
			-- time for some common replacements!
			if replace and front then
				local asplit = {}
				local ba = front:gmatch('[^\']+')
				for a in ba do
					table.insert(asplit,a)
				end
				local replaceflag = false
				if #asplit > 1 then
					local prev = asplit[#asplit-1]:lower()
					local last = asplit[#asplit]:lower()
					if last == 'll' then
						asplit[#asplit] = 'will'
						replaceflag = true
					elseif last == 'm' then
						asplit[#asplit] = 'am'
						replaceflag = true
					elseif last == 've' then
						asplit[#asplit] = 'have'
						replaceflag = true
					elseif last == 're' then
						asplit[#asplit] = 'are'
						replaceflag = true
					elseif last == 's' then
						if prev == 'he' or prev == 'she' 
							or prev == 'that' or prev == 'this'
							or prev == 'it' or prev == 'how' 
							or prev == 'why' or prev == 'who'
							or prev == 'when' or prev == 'what' then
							asplit[#asplit] = 'is'
							replaceflag = true
						end
					end
				end
				if not(replaceflag) then
					tokens[#tokens+1] = front
				else
					for i=1,#asplit do
						tokens[#tokens+1] = asplit[i]
					end
				end
			else
				tokens[1] = front
			end

			--suffic characters/punctuation to tokens
			for i=#back,1,-1 do
				tokens[#tokens+1] = back[i]
			end
		else
			tokens[1] = elem
		end
		for i,v in pairs(tokens) do table.insert(splitstring,tokens[i]) end
	end
	return splitstring
end

function data_processing(rawdata,pattern,lower,wordopt)
	local usemostcommon = false
	local useNmostcommon = 4500
	if wordopt then
		usemostcommon = wordopt.usemostcommon or false
		useNmostcommon = wordopt.useNmostcommon or 4500
	end
	local embeddings = {}
	local deembeddings = {}
	local freq = {}
	local numkeys = 0
	local numwords = 0

	-- split the string and make embeddings/deembeddings/freq
	local splitstring = split_string(rawdata,pattern,lower,wordopt)
	numwords = #splitstring
	tokenized = torch.zeros(numwords)
	for i=1,numwords do
		if not embeddings[splitstring[i]] then
			numkeys = numkeys + 1
			embeddings[splitstring[i]] = numkeys
			deembeddings[numkeys] = splitstring[i]
			freq[numkeys] = {1,numkeys}
		else
			freq[embeddings[splitstring[i]]][1] = freq[embeddings[splitstring[i]]][1] + 1
		end
		tokenized[i] = embeddings[splitstring[i]]
	end

	-- only take the most frequent entries
	local num_represented = 0
	if usemostcommon then
		numkeys = math.min(numkeys,useNmostcommon)
		table.sort(freq,function(a,b) return a[1]>b[1] end)
		local new_embed = {}
		local new_deembed = {}
		for i=1,numkeys do
			new_deembed[i] = deembeddings[freq[i][2]]
			new_embed[new_deembed[i]] = i
			num_represented = num_represented + freq[i][1]
		end
		embeddings = new_embed
		deembeddings = new_deembed
		print('Dictionary captures about ', 100*num_represented/numwords, '% of text.')
		-- rebuild tokenized:
		for i=1,numwords do
			tokenized[i] = embeddings[splitstring[i]] or numkeys + 1
		end
	end

	return embeddings, deembeddings, numkeys, numwords, tokenized, freq
end

