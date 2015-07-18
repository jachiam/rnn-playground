-- This file exists purely to allow for some manipulations on some data files I have lying around.

function data(file)
	local file = file or 'test_data.txt'
	local f = torch.DiskFile(file)
	local rawdata = f:readString('*a')
	f:close()
	return rawdata
end

function process_GSAR(rawdata,mode)
	-- mode = 1 means forum post data
	-- mode = 2 means chatroom data
	local table_entries = {}
	local speakers = {}
	local messages = {}
	local breakapart = rawdata:gmatch('%([^\n]+%),\n')	-- old pattern that mostly worked was '%b(),\n'
	local j = 0
	local broken_message = {}
	local mode = mode or 2
	local i_start = mode + 1
	for table_entry in breakapart do
		--table.insert(table_entries,table_entry)
		local breakdown = table_entry:gmatch('%b\'\'')

		broken_message = {}
		for breakpiece in breakdown do
			table.insert(broken_message,breakpiece)
		end

		local reconstructed_message = ''
		for i=i_start,#broken_message do
			reconstructed_message = reconstructed_message .. broken_message[i]
		end
		reconstructed_message = reconstructed_message:sub(2,reconstructed_message:len()-1)
		if mode == 1 then
			reconstructed_message = process_raw_post(reconstructed_message)
		else 
			reconstructed_message = process_raw_message(reconstructed_message)
		end

		if broken_message[i_start - 1] then
			local name = broken_message[i_start - 1]
			name = name:sub(2,name:len()-1)

			table.insert(speakers,name)
			table.insert(messages,reconstructed_message)
		end
		j = j + 1
		if j % 100 == 0 then
			collectgarbage()
		end
		if #messages > 100000 then
			print('Breaking now.')
			break
		end
	end
	return table_entries,speakers,messages
end

function process_write_GSAR(rawdata,mode)
	local f = torch.DiskFile('test_output.txt','rw')
	f:seekEnd()
	f:writeString('\n')

	local breakapart = rawdata:gmatch('%([^\n]+%),\n')	-- old pattern that mostly worked was '%b(),\n'
	local broken_message = {}
	local j = 0
	local mode = mode or 2
	local i_start = mode + 1

	for table_entry in breakapart do
		local breakdown = table_entry:gmatch('%b\'\'')

		broken_message = {}
		for breakpiece in breakdown do
			table.insert(broken_message,breakpiece)
		end

		local reconstructed_message = ''
		for i=i_start,#broken_message do
			reconstructed_message = reconstructed_message .. broken_message[i]
		end
		reconstructed_message = reconstructed_message:sub(2,reconstructed_message:len()-1)
		if mode == 1 then
			reconstructed_message = process_raw_post(reconstructed_message)
		else 
			reconstructed_message = process_raw_message(reconstructed_message)
		end

		if broken_message[i_start-1] then
			f:writeString(reconstructed_message .. '\n') 
		end
		j = j + 1
		if j % 1000 == 0 then
			collectgarbage()
		end
	end
	f:close()
end

function process_raw_message(msg)
	local message = msg
	message = string.gsub(message, '\\\\\'\'', '\'')
	message = string.gsub(message, '\\\\\"', '\"')
	message = string.gsub(message, '&gt;','>')
	message = string.gsub(message, '&lt;','<')
	message = string.gsub(message,'<img.*alt=\"(.+)\".*>','(%1)')
	message = string.gsub(message,'<img.*>','<img>')
	message = string.gsub(message,'<br ?/>','\n')
	message = string.gsub(message,'https?://%S+','%[link%]')
	-- Kill any non-reasonable character:
	message = string.gsub(message,'[^%w%s%p%c]','')
	return message
end

function process_raw_post(post)
	local post = post
	post = string.gsub(post,'<img.*alt=\'\'(.+)\'\'.*>','(%1)')
	post = string.gsub(post,'&#39;','\'')
	post = string.gsub(post,'&quot;','\"')
	post = string.gsub(post,'&#33;','!')
	post = string.gsub(post,'%[attachment=.-%]','')
	post = string.gsub(post,'%[quote.*%].-%[/quote%]','')
	post = string.gsub(post,'%[url.*%](.-)%[/url%]','%1')
	post = string.gsub(post,'<br ?/>\\n<br ?/>\\n','\n')
	post = string.gsub(post,'<br ?/>\\n','\n')
	-- this one gets rid of [i][/i] tags and the like
	post = string.gsub(post,'%[(.*).*%](.-)%[/%1%]','%2') 
	post = process_raw_message(post)
	return post
end

function write_test()
	local f = torch.DiskFile('test_output.txt','rw')
	f:seekEnd()
	_,_,messages = process_GSAR(rawdata)
	f:writeString('\n')
	for i=1,#messages do
		f:writeString(messages[i] .. '\n') 
		messages[i] = nil
		if i % 1000 == 0 then
			collectgarbage()
		end
	end
	f:close()
end


rawdata = data('chatroom.200k.txt')
--rawdata = data()
